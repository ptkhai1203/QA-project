import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from model import LSTMLogiQA, BLSTMLogiQA
from os.path import join
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, LogiQA


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    # LSTM model for LogiQA
    log.info('Building model...')
    # model = LSTMLogiQA(word_vectors=word_vectors,
    #                    hidden_size=args.hidden_size)
    model = BLSTMLogiQA(word_vectors=word_vectors,
                        hidden_size=args.hidden_size)
    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    print(record_file)
    dataset = LogiQA(record_file)
    # filter data without no answer
    print('original len: ', len(dataset))
    #def filter_fn(dataset):
    #    res = []
    #    for x in dataset:
    #        if x[4] != 0 and x[5] != 0:
    #          res.append(x)
    #    return res
    #dataset = filter_fn(dataset)
    print('len after filter: ', len(dataset))
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    print(eval_file)
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, qw_idxs, op_idxs, labels, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            op_idxs = op_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_out = model(cw_idxs, qw_idxs, op_idxs)
            label = labels.to(device)
            loss = F.nll_loss(log_out, label)
            nll_meter.update(loss.item(), batch_size)

            # Get ACC score
            p1 = log_out.exp()
            print(p1)
            _, out_pred = torch.max(p1,dim=1)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            preds = util.convert_tokens(ids.tolist(),
                                           out_pred)
            pred_dict.update(preds)
            sub_dict.update(preds)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict)
        results_list = [('NLL', nll_meter.avg),
                        ('ACC', results['ACC'])]
        
        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')


    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())