# import logging
# import os
# import queue
# import re
# import shutil
# import string
# import torch
# import torch.nn.functional as F
# import torch.utils.data as data
# import tqdm
# import numpy as np
# import ujson as json
# import csv
# from collections import Counter
import json
class LogiQAProcessor():
    def _read_txt(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()

        return lines
    def _create_examples(self, lines, type):
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        assert len(lines) % 8 == 0, 'len(lines)={}'.format(len(lines))
        n_examples = int(len(lines) / 8)
        examples = []

        for i in range(n_examples):
            label_str = lines[i * 8 + 1].strip()
            context = lines[i*8+2].strip()
            question = lines[i*8+3].strip()
            answers = lines[i*8+4 : i*8+8]

            examples.append({"context": context, "question": question, "answer": [item.strip()[2:] for item in answers], "label": label_map[label_str]})
        
        return {"data": examples}
    
    def get_train(self, data_dir):
        return self._create_examples(self._read_txt(data_dir), type="train")

logiQA_pro = LogiQAProcessor()
logiqa_dataset = logiQA_pro.get_train("logiqa_data/Eval.txt")
with open("logiqa_eval.json", 'w') as f:
    json.dump(logiqa_dataset, f, indent=4)
# class LogiQA(data.Dataset):
#     def __init__(self, data_path):
#         super(LogiQA, self).__init__()
#         dataset = np.load(data_path)
#         self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
#         self.question_idxs = torch.from_numpy(dataset['question_idxs']).long()
#         self.option1 = torch.from_numpy(dataset['option'])