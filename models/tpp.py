import argparse
import numpy as np
import pickle
import time
import os
import random

import constants
# 
# from ..dataset import get_dataloader
# from transformer.Models import Transformer
# from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from torch.distributions import Categorical


import matplotlib.pyplot as plt


# from stat_funcs import mad, get_time_scale
from .eemb import TimeEncode

class TPP(nn.Module):
    def __init__(self, d_model, marker_num, device):
        super().__init__()
        self.device = device
        
        self.itv_emb = TimeEncode(d_model) # time interval embedder
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(),
            nn.Linear(d_model, marker_num),
            nn.Softplus()
        )

    def intensity(self, h, dt):
        """h: (b, s, d), dt: (b, s) -> intensity: (b, s, d)"""
        return self.mlp(h + self.itv_emb(dt))
    
    def total_intensity(self, h, dt):
        return self.intensity(h, dt).sum(-1, keepdim=True)
        
    def total_intensity_measure(self, h, dt, num_samples=10):
        """
        Monte-Carlo integration (uniform)

        h: (..., H)
        dt: (...,)
        """
        intvls = dt.unsqueeze(-1) # (..., 1) intervals
        intvls = intvls * torch.linspace(0, 1, num_samples).to(h.device) # (..., N)
        
        hids = repeat(h, "... H -> ... N H", N=num_samples)
        intn_samples = self.total_intensity(hids, intvls) # (..., N, 1)
        intn_samples = intn_samples.squeeze(-1) # (..., N)
        intn_mesr = intn_samples.mean(-1, keepdims=True) * dt.unsqueeze(-1) # (..., 1), e.g. (B, S, 1)
        return intn_mesr
        
    def log_prob(self, h, dt, markers):
        """return ll (B, S - 1, 1)"""
        # padding mask
        pad_mask = (markers == constants.PAD).to(self.device)     
        
        # intn
        intn_s = self.intensity(h, dt)
        
        # intn_k
        real_markers = markers - 1 # 抵消预处理时marker+1
        real_markers[pad_mask] = 0
        index = real_markers.unsqueeze(-1)
        intn_ms = intn_s.gather(dim=-1, index=index) # (B, S - 1, 1)
        
        # log_intn_ms
        log_intn_ms = intn_ms.log()
        
        # intn_mesr
        intn_mesr = self.total_intensity_measure(h, dt)
        
        return log_intn_ms - intn_mesr

    
    def loss(self, h, dt, markers, mask=None):
        ll = self.log_prob(h, dt, markers)
        ll = ll[~mask].mean() if mask is not None else ll.mean()
        return ll
    