import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
from . import BasicModule
from module import Embedding
from module import GCN as GCNBlock
from utils import seq_len_to_mask


class GCN(BasicModule):
    def __init__(self, cfg):
        super(GCN, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.input_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.input_size = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.gcn = GCNBlock(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)

    def forward(self, x):
        word, lens, head_pos, tail_pos, adj = x['word'], x['lens'], x['head_pos'], x['tail_pos'], x['adj']


        inputs = self.embedding(word, head_pos, tail_pos)
        output = self.gcn(inputs, adj)
        output = output.max(dim=1)[0]
        output = self.fc(output)

        return output
