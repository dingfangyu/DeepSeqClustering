import argparse
import numpy as np
import pickle
import time
import random

from dataset import get_dataloader
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import repeat, rearrange
from torch.distributions import Categorical

def set_seed(seed_val=0):
    
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-device', type=int, default=0)

    parser.add_argument('-n_comps', type=int, default=4)
    parser.add_argument('-agg', type=str, default='attn')

    parser.add_argument('-model', type=str, default='cstpp')
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-epoch', type=int, default=30)
    

    return parser


def mad(arr: torch.tensor, n=5):
    """
    median absolute deviation
    """
    median = torch.median(arr)
    deviations = torch.abs(arr - median)
    mad = torch.median(deviations)

    indices = torch.where(torch.abs(arr - median) < n * mad)[0]

    return indices


def get_time_scale(trainloader):
    l = []
    for t, i, m, _ in trainloader:
        i = i[m != 0]
        l.append(i[mad(i)].mean())
    return torch.stack(l).mean()



def get_last_event_info(h, times, z, pad_mask, K):
    """
    find the last event in the same cluster
    """
    clusters = z.argmax(-1)
    clusters_indices = torch.nonzero(clusters == rearrange(torch.arange(K), '(k a b) -> k a b ', a=1, b=1), as_tuple=False) # k_id, b_id, s_id
    
    is_first_event = (clusters_indices[1:, 0] != clusters_indices[:-1, 0]) | (clusters_indices[1:, 1] != clusters_indices[:-1, 1]) 
    is_first_event = torch.cat([torch.tensor([True]).to(is_first_event.device), is_first_event])
    
    clusters_indices = clusters_indices[:, 1:] # (b*s, [b_id, s_id])
    
    first_event_mask = torch.zeros_like(pad_mask, device=pad_mask.device)
    first_event_mask[clusters_indices[:, 0], clusters_indices[:, 1]] = is_first_event
    
    last_event_idx = clusters_indices[:-1]
    last_event_idx = torch.cat([torch.tensor([[-1, -1]]).to(last_event_idx.device), last_event_idx])
    last_event_idx[is_first_event] = torch.tensor([[-1, -1]]).to(last_event_idx.device)
    
    h_last = h.clone()
    h_last[clusters_indices[:, 0], clusters_indices[:, 1]] = h[last_event_idx[:, 0], last_event_idx[:, 1]] 
    
    t_last = times.clone()
    t_last[clusters_indices[:, 0], clusters_indices[:, 1]] = times[last_event_idx[:, 0], last_event_idx[:, 1]] 
    dt_last = times - t_last
    
    return h_last, dt_last, first_event_mask

def rearrange_zi(z, opt):
    idx_ks = []
    for k in range(opt.n_comps):
        idx_k = torch.where(z == k)
        idx_ks.append(idx_k)

    def key_func(x):
        return x[0][0] if len(x[0]) else 99999
    sorted_idx = sorted(idx_ks, key=key_func)
    
    for k in range(opt.n_comps):
        z[sorted_idx[k]] = k
        
    return z

def rearrange_z(z, opt):
    for i in range(z.shape[0]):
        zi = rearrange_zi(z[i], opt)
        z[i] = zi
    return z