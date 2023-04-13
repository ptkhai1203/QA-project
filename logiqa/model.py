import layers
import torch
import torch.nn as nn

class LSTMLogiQA(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(LSTMLogiQA, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors, 
                                    hidden_size=hidden_size, 
                                    drop_prob=drop_prob)
        self.enc = layers.LSTMEncoder(input_size=hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      drop_prob=drop_prob)
        self.out = layers.OutputLogiQA(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, op_idxs):
        x = torch.concat([cw_idxs, qw_idxs, op_idxs.view(cw_idxs.size(0), -1)], dim=1)
        x_mask = torch.zeros_like(x) != x
        x_len = x_mask.sum(-1)
        emb_x = self.emb(x)
        enc_x = self.enc(emb_x, x_len)
        out = self.out(enc_x, x_mask)

        return out

class BLSTMLogiQA(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BLSTMLogiQA, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors, 
                                    hidden_size=hidden_size, 
                                    drop_prob=drop_prob)
        self.enc = layers.BLSTMEncoder(input_size=hidden_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      drop_prob=drop_prob)
        self.out = layers.OutputLogiQABLSTM(hidden_size=hidden_size)

    def forward(self, cw_idxs, qw_idxs, op_idxs):
        x = torch.concat([cw_idxs, qw_idxs, op_idxs.view(cw_idxs.size(0), -1)], dim=1)
        x_mask = torch.zeros_like(x) != x
        x_len = x_mask.sum(-1)
        emb_x = self.emb(x)
        enc_x = self.enc(emb_x, x_len)
        out = self.out(enc_x, x_mask)

        return out
