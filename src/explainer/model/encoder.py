"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from ..blocks.encoder_layer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        
        # Encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                ffn_hidden=ffn_hidden,
                                                n_head=n_head,
                                                drop_prob=drop_prob)
                                    for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x