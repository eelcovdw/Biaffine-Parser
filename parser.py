import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import h5py

from modules import MLP, BiAffineAttn, TimeDistributed

"""
PyTorch implementation of the BiAffine Dependency Parser, as defined in

Dozat, T., & Manning, C. D. (2016). Deep biaffine attention for neural dependency parsing.
https://arxiv.org/abs/1611.01734
"""

class BiAffineParser(nn.Module):
    def __init__(self, word_vocab_size, word_emb_dim, 
                 pos_vocab_size, pos_emb_dim, emb_dropout,
                 lstm_hidden, lstm_depth, lstm_dropout,
                 arc_hidden, arc_depth, arc_dropout, arc_activation,
                 lab_hidden, lab_depth, lab_dropout, lab_activation,
                 n_labels):
        super(BiAffineParser, self).__init__()

        # Embeddings
        self.word_embedding = TimeDistributed(nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0))
        self.pos_embedding = TimeDistributed(nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0))
        self.emb_dropout = nn.Dropout(p=emb_dropout)


        # LSTM
        lstm_input = word_emb_dim + pos_emb_dim
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden, 
                            batch_first=True, dropout=lstm_dropout, bidirectional=True)

        # MLPs
        self.arc_mlp_h = TimeDistributed(MLP(lstm_hidden*2, arc_hidden, arc_depth, 
                                             arc_activation, arc_dropout))
        self.arc_mlp_d = TimeDistributed(MLP(lstm_hidden*2, arc_hidden, arc_depth, 
                                             arc_activation, arc_dropout))
                                             
        self.lab_mlp_h = TimeDistributed(MLP(lstm_hidden*2, lab_hidden, lab_depth,
                                             lab_activation, lab_dropout))
        self.lab_mlp_d = TimeDistributed(MLP(lstm_hidden*2, lab_hidden, lab_depth,
                                             lab_activation, lab_dropout))    

        # BiAffine layers
        self.arc_attn = BiAffineAttn(arc_hidden, 1, bias_head=False, bias_dep=True)
        self.lab_attn = BiAffineAttn(lab_hidden, n_labels, bias_head=True, bias_dep=True)

        
    def forward(self, x_word, x_pos, lengths=None):
        # Embeddings
        x_word = self.word_embedding(x_word)
        x_pos = self.pos_embedding(x_pos)
        x = torch.cat([x_word, x_pos], dim=-1)
        x = self.emb_dropout(x)

        # LSTM
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        # MLPs
        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)

        # Attention
        S_arc = self.arc_attn(arc_h, arc_d)
        S_lab = self.lab_attn(lab_h, lab_d)

        return S_arc, S_lab

    def load_word_embedding(self, filename):
        """
        Function to load external word embeddings.
        """
        with h5py.File(filename, 'r') as f:
            embeddings = torch.from_numpy(f['word_vectors'][:])
        if not embeddings.shape == self.word_embedding.module.weight.size():
            raise RuntimeError("embedding layer shape does not match pretrained embedding ({}, {})".format(
                embeddings.shape, self.word_embedding.module.weight.size()))
        self.word_embedding.module.weight = nn.Parameter(embeddings)

    def get_param_groups(self):
        """
        Returns parameters in groups base (embeddings + lstm), arc (arc MLPs and attention),
        and label (label MLPs and attention)
        """
        base = []
        arc = []
        label = []
        for p_name, param in self.named_parameters():
            if 'arc' in p_name:
                arc.append(param)
            elif 'lab' in p_name:
                label.append(param)
            else:
                base.append(param)
        return base, arc, label
