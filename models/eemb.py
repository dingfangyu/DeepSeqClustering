import torch
from torch import nn
import constants

class TimeEncode(nn.Module): 
    # ref: TGAT
    def __init__(self, d_model):
        super().__init__()
        self.basis_freq = nn.Parameter(1 / 10 ** torch.linspace(0, 9, d_model).float())
        self.phase = nn.Parameter(torch.zeros(d_model).float())
        
    def forward(self, ts):
        # ts: (*, )
        ts = ts.unsqueeze(-1) # [*, 1]
        map_ts = ts * self.basis_freq + self.phase
        harmonic = torch.cos(map_ts)
        return harmonic 
    
class EventEmb(nn.Module):
    def __init__(self, d_model, marker_num):
        super().__init__()
        self.time_emb = TimeEncode(d_model)
        self.marker_emb = nn.Embedding(marker_num + 1, d_model, padding_idx=constants.PAD) # pad=-1
        
    def forward(self, times, markers):
        return self.time_emb(times) + self.marker_emb(markers)