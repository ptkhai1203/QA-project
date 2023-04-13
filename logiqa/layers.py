import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

class Embedding(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)

    def forward(self, x):
        emb = self.embed(x) # (batch_size, x_len, vec_dim)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb) # (batch_size, x_len, hidden_size)

        return emb
    
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.):
        super(LSTMEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout = drop_prob if drop_prob > 0 else 0)
        
    def forward(self, x, lengths):
        # x size: batch_size, x_len, hidden_size
        orig_len = x.size(1)
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, (hn, cn) = self.lstm(x) # hn size: 1, batch_size, hidden_size
        return hn[-1]

class BLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.):
        super(BLSTMEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout = drop_prob if drop_prob > 0 else 0)
        
    def forward(self, x, lengths):
        # x size: batch_size, x_len, hidden_size
        orig_len = x.size(1)
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, (hn, cn) = self.lstm(x) # hn size: 1, batch_size, 2*hidden_size
        return hn[-1]

class OutputLogiQABLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(OutputLogiQABLSTM, self).__init__()
        self.w = nn.Linear(hidden_size, 4)

    def forward(self, x, mask):
        out1 = self.w(x) # batch_size, 4
        logits = torch.nn.functional.log_softmax(out1, dim=1) # batch_size, 4
        return logits

class OutputLogiQA(nn.Module):
    def __init__(self, hidden_size):
        super(OutputLogiQA, self).__init__()
        self.w = nn.Linear(hidden_size, 4)

    def forward(self, x, mask):
        out1 = self.w(x) # batch_size, 4
        logits = torch.nn.functional.log_softmax(out1, dim=1) # batch_size, 4
        return logits

