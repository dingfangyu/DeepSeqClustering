import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import rearrange

# normal self-attention seq2seq module
class MultiHeadSelfAttn(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, n_head):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_head = n_head
        
        # we assume Q/K/V have same shapes
        self.W_Q = nn.Linear(inp_dim, hid_dim * n_head)
        self.W_K = nn.Linear(inp_dim, hid_dim * n_head)
        self.W_V = nn.Linear(inp_dim, hid_dim * n_head)
        
        self.W_O = nn.Linear(hid_dim * n_head, out_dim)
        
    def forward(self, x, pad_mask=None, attn_mask=None):
        # x: (batched) sequential variable, shape: (..., seq_len, feat_dim)
        # batch_first=True is the default setting
        # dropout is omitted in our primary implemetation
        
        # pad_mask: (b, s)
        # attn_mask: (s, s)
        
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x) # (..., s, hid_dim * n_head)
        Q = rearrange(Q, '... s (h n) -> ... n s h', n=self.n_head)
        K = rearrange(K, '... s (h n) -> ... n s h', n=self.n_head)
        V = rearrange(V, '... s (h n) -> ... n s h', n=self.n_head) # (..., n, s, h)
        
        score = Q @ rearrange(K, '... s h -> ... h s') # QK^T (..., n, s, s)
        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, 'b (n s a) -> b n s a', n=1, a=1) # (b, n, s, 1)
            score = score.masked_fill(pad_mask, -1e26) # mask where the mask value == True
        if attn_mask is not None:
            score = score.masked_fill(attn_mask, -1e26)
                    
        attn = F.softmax(score / (self.hid_dim ** 0.5), dim=-1) # (..., n, s, s)
        out = attn @ V # (..., n, s, h)
        out = rearrange(out, '... n s h -> ... s (n h)')
        out = self.W_O(out)
        
        return attn, out
    

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        return self.mlp(x)

class TfmEncLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.mhsa = MultiHeadSelfAttn(d_model, d_model, d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.ln1(x + self.mhsa(x, pad_mask, attn_mask)[1])
        x = self.ln2(x + self.ffn(x))
        return x

    
class TfmEnc(nn.Module):
    def __init__(self, n_layer, layer, **layer_params):
        super().__init__()
        self.layers = nn.ModuleList(layer(**layer_params) for i in range(n_layer))
    def forward(self, x, pad_mask, attn_mask):
        for layer in self.layers:
            x = layer(x, pad_mask, attn_mask)
        return x