"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from ..embedding.positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):

    def __init__(self, d_model, n_head, seq_len, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model=d_model,
                                            max_len=max_len,
                                            device=device)

        self.dropout = nn.Dropout(p=drop_prob)
        
        # Encoder
        self.encoder = Encoder(d_model=d_model,
                             n_head=n_head,
                             max_len=max_len,
                             ffn_hidden=ffn_hidden,
                             enc_voc_size=d_model, 
                             drop_prob=drop_prob,
                             n_layers=n_layers,
                             device=device)

        #self.sequence_weights = nn.Parameter(torch.ones(d_model))
        self.proj_layer = nn.Linear(seq_len, 1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        pos_encoding = self.pos_encoder(x)
        x_ = x + pos_encoding
        # print('x_',x_.max(),x_.min())
        x_ = self.dropout(x_)
        # print('x_',x_.max(),x_.min())
        # 3. 通过encoder
        enc_output = self.encoder(x_, mask)  # [batch_size, seq_len, d_model]

        # print(enc_output.shape)
        weights = self.proj_layer(enc_output.transpose(-2,-1)).squeeze(-1).unsqueeze(1) # [batch_size, 1, d_model]

        enc_output = torch.sum(x * weights, dim=-1) # [batch_size, seq_len]
        enc_output = enc_output.unsqueeze(-1).repeat(1,1,d_model)

        min_output = torch.amin(enc_output,keepdim=True)
        enc_output = enc_output - min_output
        # print('min_output',enc_output.max(),enc_output.min())
        enc_output = enc_output / (torch.amax(enc_output,keepdim=True)+1e-13)

        return enc_output