# EventSeqDecomp

### Introduction

An **unsupervised event sequence decomposition** method is proposed to mine highly correlated subsequences (clusters) with:

- neural temporal point process 
- sequential variational autoencoder
- TGAT time encoding 
- NRI decoder 
- beta-VAE loss 

Please refer to [Google Drive link](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing) for real-world event sequence datasets.

### Related Works

- Deep Variational Information Bottleneck (ICLR 2017)
- Neural Relational Inference for Interacting Systems (ICML 2018)
- Inductive Representation Learning on Temporal Graphs (ICLR 2020)
- Transformer Hawkes Process (ICML 2020)
- c-NTPP: Learning Cluster-Aware Neural Temporal Point Process (AAAI 2023, **ours**)

### Methodology

In this repo we implement modules:

- neural event embedding ([models/eemb.py](models/eemb.py))
- neural temporal point process ([models/tpp.py](models/tpp.py))
- cluster aware self attention & transformer ([models/ctfm.py](models/ctfm.py))

Pseudo code for the sequential variational autoencoder:

```python
# X: input sequence with shape (length, embedding_dim)
# q: posterior distribution parameter with shape (length, cluster_num)
# Z: sequence-decomposition 1-of-K categorical sampling result
# h: history embedding for each event with shape (length, hidden_dim)

# --- Encoder (Inference)
q = TfmEnc(X) # transformer encoder
Z = gumble_softmax(q, hard=True)

# --- Decoder (Reconstruction)
h = cTfmEnc(X, Z) # cluster(Z)-aware transformer encoder
likelihood = TPP(X, h) # temporal point process model with neural intensity
kld = KLD(q, prior)

elbo = likelihood - kld * beta
```

We also develop a differentiable **cluster aware self attention** module, by utilizing the categorical variable **Z**, we modify the classic self attention module to compute the representation for each event based on the events in the same cluster:

```python
# input: X, Z
Q, K, V = W_Q(X), W_K(X), W_V(X)
out = norm(Z @ Z.T * softmax(QK^T / \sqrt{d})) @ V
```

### Requirements

```
einops==0.4.1
matplotlib==3.3.4
numpy==1.19.5
torch==1.8.2+pai
tqdm==4.64.0
```

### Reference

```
@coming_soon
```
