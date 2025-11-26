# NLL loss for survival analysis

import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import torch.nn.functional as F
import collections
from torch.utils.data.dataloader import default_collate


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):

    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards

    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0], h[y] = h(1), S[1] = S(1)

    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
                                  + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


# def l1_reg_all(model, reg_type=None):
#     l1_reg = None

#     for W in model.parameters():
#         if l1_reg is None:
#             l1_reg = torch.abs(W).sum()
#         else:
#             l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
#     return l1_reg


# def l1_reg_modules(model, reg_type=None):
#     l1_reg = 0

#     l1_reg += l1_reg_all(model.fc_omic)
#     l1_reg += l1_reg_all(model.mm)

#     return l1_reg
