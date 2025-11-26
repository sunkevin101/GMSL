import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from nystrom_attention import Nystromformer, NystromAttention


class Transformer_Encoder_Cls(nn.Module):
    def __init__(self, dim, nhead=8, num_layers=1, dropout=0.25, add_cls=True):
        super(Transformer_Encoder_Cls, self).__init__()
        """
        default: 8 head, 1 layer
        """
        self.add_cls = add_cls

        self.encoder = Nystromformer(dim=dim,
                                     depth=num_layers,
                                     dim_head=dim // 8,
                                     heads=nhead,
                                     num_landmarks=dim // 2,
                                     attn_values_residual=True,
                                     attn_dropout=dropout,
                                     ff_dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):

        if self.add_cls:
            x = torch.cat([self.cls_token, x], dim=1)  # add Cls token
            x = self.encoder(x)  

            return x[:, 0], x[:, 1:] 

        else:
            x = self.encoder(x) 

            return x  
