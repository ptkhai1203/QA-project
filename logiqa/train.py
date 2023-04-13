import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from model import LSTMLogiQA, BLSTMLogiQA
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, LogiQA

def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    # tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    # BiDAF model
    print(type(word_vectors))
    print(word_vectors.size())
    # model = LSTMLogiQA(word_vectors=word_vectors,
    #                    hidden_size=args.hidden_size,
    #                    drop_prob=args.drop_prob)
    model = BLSTMLogiQA(word_vectors=word_vectors,
                        hidden_size=args.hidden_size,
                        drop_prob=args.drop_prob)
    # Base LSTM model
    #model = LSTMQA(word_vectors=word_vectors,
    #              hidden_size=args.hidden_size,
    #              drop_prob=args.drop_prob)
    # BLSTM model
    #model = BLSTMQA(word_vectors=word_vectors,
    #                hidden_size=args.hidden_size,
    #                drop_prob=args.drop_prob)
    # ALSTM model
    #model = ALSTMQA(word_vectors=word_vectors,
    #                hidden_size=args.hidden_size,
    #                drop_prob=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = LogiQA(args.train_record_file)
    print('original len: ', len(train_dataset))
    indices = torch.arange(min(20000, len(train_dataset)))
    train_k = torch.utils.data.Subset(train_dataset, indices)
    train_loader = data.DataLoader(train_k,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = LogiQA(args.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, qw_idxs, op_idxs, labels, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                # print(cw_idxs.size())
                log_out = model(cw_idxs, qw_idxs, op_idxs)
                label = labels.to(device)
                # print(log_out, label)
                loss = F.nll_loss(log_out, label)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                # tbx.add_scalar('train/NLL', loss_val, step)
                # tbx.add_scalar('train/LR',
                              #  optimizer.param_groups[0]['lr'],
                              #  step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')



def evaluate(model, data_loader, device, eval_file, max_len):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
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
            _, out_pred = torch.max(p1,dim=1)
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds = util.convert_tokens(ids.tolist(),
                                           out_pred)
            pred_dict.update(preds)

    model.train()
    results = util.eval_dicts(gold_dict, pred_dict)
    results_list = [('NLL', nll_meter.avg),
                    ('ACC', results['ACC'])]
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())