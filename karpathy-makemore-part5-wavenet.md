# makemore Part 5: Building a WaveNet

**Source**: Andrej Karpathy's "makemore" series, Part 5
**Prerequisites**: Parts 1–4 (MLP, BatchNorm, manual backprop)
**Theme**: Scaling up from a flat MLP to a hierarchical, deeper architecture inspired by WaveNet

---

## Overview

This lecture moves from the flat single-hidden-layer MLP to a **deeper, hierarchical architecture** that progressively fuses character information — inspired by DeepMind's WaveNet (2016). Instead of crushing all input characters into one layer immediately, we fuse them in pairs: first bigrams, then pairs of bigrams, and so on in a tree structure. Along the way, we PyTorch-ify the code further (custom `Embedding`, `FlattenConsecutive`, `Sequential` modules), fix a BatchNorm bug for 3D inputs, and discuss how this architecture relates to dilated causal convolutions.

**Performance progression**:

| Configuration | Val Loss |
| --- | --- |
| Flat MLP, block_size=3 (Part 3) | 2.10 |
| Flat MLP, block_size=8 | 2.027 |
| Hierarchical, block_size=8, same params | 2.029 → 2.022 (after BN fix) |
| Hierarchical, scaled up (76K params) | **1.993** |

---

## 1. Starting Point and Housekeeping

### 1.1 Starter Code

The code picks up from Part 3 (not Part 4, which was the backprop exercise). We have the same data pipeline: 182,000 examples of `block_size` characters predicting the next character, split into train/dev/test.

### 1.2 Fixing the Noisy Loss Plot

The training loss curve is noisy because mini-batches of 32 are small. Fix: reshape the flat list of losses into rows of 1000 and take the mean of each row.

```python
# Before: jagged, unreadable
plt.plot(lossi)

# After: smooth, informative
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
```

```
Before:                          After:
 ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌          ──╲
 ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌              ╲──────
 (unreadable noise)                    ╲────── (learning rate decay)
                                          ────── (converged)
```

**How it works**: `view(-1, 1000)` reshapes e.g. 200,000 loss values into a (200, 1000) matrix. `.mean(1)` averages each row, giving 200 smooth data points. This is just a visualization trick — the training itself is unchanged.

---

## 2. PyTorch-ifying: New Modules

### 2.1 Embedding Module

Previously, the embedding table `C` was a special case outside the layer list. Now it becomes a proper module:

```python
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]    # index lookup
        return self.out

    def parameters(self):
        return [self.weight]
```

```
Forward:
  IX (32, 8):  integer tensor, each entry is a character index (0–26)
  C  (27, 10): embedding table, each row is a 10-dim vector

  emb = C[IX] → (32, 8, 10)

  IX:                      C:                        emb:
  [[0, 5, 13, ...],       row 0: [0.5, 0.1, ...]    emb[0,0] = C[0]
   [5, 13, 13, ...],      row 1: [0.3, 0.8, ...]    emb[0,1] = C[5]
   ...]                    ...                        emb[0,2] = C[13]
                           row 26: [0.2, 0.9, ...]   ...
```

Mirrors `torch.nn.Embedding`.

### 2.2 Flatten Module

Previously, `emb.view(B, -1)` was done manually. Now it's a module. But we'll need to evolve it — see section 4.

```python
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []
```

Mirrors `torch.nn.Flatten`.

### 2.3 Sequential Container

Instead of a naked list of layers, wrap them in a `Sequential` module:

```python
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

```python
# Before:
layers = [Embedding(...), Flatten(), Linear(...), BatchNorm1d(...), Tanh(), ...]
parameters = [p for layer in layers for p in layer.parameters()]
for layer in layers:
    x = layer(x)

# After:
model = Sequential([Embedding(...), Flatten(), Linear(...), ...])
parameters = model.parameters()
logits = model(xb)          # one clean call
```

Mirrors `torch.nn.Sequential`.

### 2.4 Simplified Forward Pass

```python
# Before (5+ lines):
emb = C[Xb]
x = emb.view(emb.shape[0], -1)
for layer in layers:
    x = layer(x)
logits = x

# After (1 line):
logits = model(Xb)
```

### 2.5 The BatchNorm Training Mode Bug (Again)

When sampling from the model, we must set `training = False` on all layers. Forgetting this causes BatchNorm to use batch statistics instead of running statistics. With a single example (batch size = 1), the variance of a single number is NaN:

```python
torch.tensor([5.0]).var()    # → NaN (variance of 1 number is undefined)
```

This silently corrupts all downstream computation. The loss may still compute (just wrong), making the bug hard to detect.

---

## 3. The Problem with Flat Architecture

### 3.1 Why Not Just Add More Layers?

The current architecture crushes all 8 characters into a single layer immediately:

```
8 chars × 10 dims = 80 numbers ──→ 200 hidden ──→ 27 output
                    ↑
             all information squashed in one step
```

We could add more hidden layers after this, but the damage is done — 80 numbers of raw character info are fused immediately into 200 neurons. This is wasteful.

### 3.2 The WaveNet Inspiration

The WaveNet paper (DeepMind, 2016) uses a **hierarchical tree structure**:

```
Flat (current):                    Hierarchical (WaveNet-like):

chars: 1  2  3  4  5  6  7  8     chars: 1  2  3  4  5  6  7  8
       └──┴──┴──┴──┴──┴──┴──┘            └──┘  └──┘  └──┘  └──┘
                 │                        bigrams: fuse pairs
          hidden layer                      └────┘    └────┘
                 │                          4-grams: fuse pairs
              output                           └────────┘
                                               8-gram: fuse
                                                  │
                                               output
```

At each level, only **two consecutive elements** are fused. Information flows gradually through the network instead of being crushed all at once.

### 3.3 Baseline: Just Scaling Up Context

Before implementing the hierarchical model, increase block_size from 3 to 8 with the flat architecture as a baseline:

| Model | Params | Val Loss |
| --- | --- | --- |
| Flat, block_size=3 | ~12K | 2.10 |
| Flat, block_size=8 | ~22K | 2.027 |

More context helps, even with the naive flat architecture.

---

## 4. Implementing Hierarchical Fusion

### 4.1 The Key Insight: Matrix Multiply Works on Higher Dimensions

PyTorch's `@` operator doesn't require 2D inputs. It operates on the **last dimension** and treats all earlier dimensions as batch dimensions:

```python
# 2D: normal matmul
(4, 80) @ (80, 200) → (4, 200)

# 3D: the first dimension is treated as batch
(4, 5, 80) @ (80, 200) → (4, 5, 200)
#  ↑ batch    ↑ batch       ↑       ↑
#             dimension     batch   matmul result

# You can have as many leading batch dimensions as you want:
(2, 3, 4, 80) @ (80, 200) → (2, 3, 4, 200)
```

```
Standard 2D matmul:
  input (4, 80)  ×  W (80, 200)  →  output (4, 200)
        ↑ batch      ↑ features

3D matmul (what we want):
  input (4, 4, 20)  ×  W (20, 200)  →  output (4, 4, 200)
        ↑ batch                               ↑ batch
           ↑ groups of 2 chars                    ↑ groups
              ↑ 2 chars × 10 dims                    ↑ hidden
```

**This means**: instead of flattening all 8 characters into 80 numbers, we can group them into 4 pairs of 2, flatten each pair into 20 numbers, and process all 4 groups **in parallel** as a batch dimension.

### 4.2 FlattenConsecutive Module

The key new module. Instead of flattening everything, it groups `n` consecutive elements:

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)    # remove spurious dimension
        self.out = x
        return self.out

    def parameters(self):
        return []
```

```
Input: emb (4, 8, 10)  ← 4 examples, 8 chars, 10-dim embeddings

FlattenConsecutive(2):
  view(4, 8//2, 10*2) = view(4, 4, 20)

  Before:  [c₁, c₂, c₃, c₄, c₅, c₆, c₇, c₈]   each cᵢ is 10-dim
  After:   [c₁c₂,  c₃c₄,  c₅c₆,  c₇c₈]           each pair is 20-dim
             ↑       ↑       ↑       ↑
           group 0  group 1  group 2  group 3

  Shape: (4, 4, 20)
         ↑  ↑   ↑
         B  groups  features (2 chars × 10 dims)
```

**Why `view` works instead of explicit `cat`**: PyTorch stores tensors in row-major (C-contiguous) order. When you `view(4, 4, 20)` a (4, 8, 10) tensor, consecutive elements in memory naturally pair up — `c₁`'s 10 values followed by `c₂`'s 10 values end up as the first group of 20. No data is copied or rearranged.

```python
# These are identical:
explicit = torch.cat([emb[:, ::2, :], emb[:, 1::2, :]], dim=2)   # (4, 4, 20)
implicit = emb.view(4, 4, 20)                                     # (4, 4, 20)
torch.all(explicit == implicit)   # → True
```

**The `squeeze` at the end**: if `T // n == 1` (all characters consumed), the middle dimension is 1, creating shape `(B, 1, C*n)`. We squeeze it to `(B, C*n)` so the final linear layer works as expected (2D input for the output logits).

### 4.3 The Hierarchical Architecture

```python
model = Sequential([
    Embedding(vocab_size, n_embd),           # (B, 8, 10)

    FlattenConsecutive(2),                    # (B, 4, 20)    ← pairs
    Linear(n_embd * 2, n_hidden, bias=False),# (B, 4, 68)
    BatchNorm1d(n_hidden),                   # (B, 4, 68)
    Tanh(),                                  # (B, 4, 68)

    FlattenConsecutive(2),                    # (B, 2, 136)   ← pairs of pairs
    Linear(n_hidden * 2, n_hidden, bias=False),# (B, 2, 68)
    BatchNorm1d(n_hidden),                   # (B, 2, 68)
    Tanh(),                                  # (B, 2, 68)

    FlattenConsecutive(2),                    # (B, 68)       ← squeezed (2//2=1)
    Linear(n_hidden * 2, n_hidden, bias=False),# (B, 68)
    BatchNorm1d(n_hidden),                   # (B, 68)
    Tanh(),                                  # (B, 68)

    Linear(n_hidden, vocab_size),             # (B, 27)       ← output logits
])
```

```
Shape flow through the network (B=32):

Xb               (32, 8)          ← integer character indices
    │ Embedding
emb              (32, 8, 10)      ← 10-dim vectors per character
    │ FlattenConsecutive(2)
                 (32, 4, 20)      ← pairs fused: 8 chars → 4 bigrams
    │ Linear(20 → 68)
                 (32, 4, 68)      ← projected to hidden dim
    │ BatchNorm + Tanh
                 (32, 4, 68)
    │ FlattenConsecutive(2)
                 (32, 2, 136)     ← bigrams fused: 4 → 2 four-grams
    │ Linear(136 → 68)
                 (32, 2, 68)      ← projected back to hidden dim
    │ BatchNorm + Tanh
                 (32, 2, 68)
    │ FlattenConsecutive(2)
                 (32, 68)         ← all fused: 2 → 1 (squeezed)
    │ Linear(136 → 68)
                 (32, 68)
    │ BatchNorm + Tanh
                 (32, 68)
    │ Linear(68 → 27)
logits           (32, 27)         ← one prediction per example

Tree structure (one example):
  chars:     c₁   c₂   c₃   c₄   c₅   c₆   c₇   c₈
              └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
  layer 1:    bigram₁   bigram₂   bigram₃   bigram₄    (4 × 68)
                └────┬────┘         └────┬────┘
  layer 2:      4-gram₁             4-gram₂             (2 × 68)
                   └──────────┬──────────┘
  layer 3:                8-gram                         (68)
                             │
  output:                  logits                        (27)
```

---

## 5. Fixing BatchNorm for 3D Inputs

### 5.1 The Bug

The original `BatchNorm1d` only reduced over dimension 0 (the batch dimension):

```python
# Original code:
xmean = x.mean(0, keepdim=True)    # reduces over dim 0 only
```

With 2D input `(32, 68)` this is correct — average over 32 examples for each of 68 channels.

But with 3D input `(32, 4, 68)`, reducing only over dim 0 gives shape `(1, 4, 68)` — maintaining 4×68 = 272 separate statistics instead of just 68. Each of the 4 group positions gets its own mean/variance, defeating the purpose of BatchNorm.

```
Wrong (reduce dim 0 only):               Correct (reduce dims 0 and 1):

Input: (32, 4, 68)                        Input: (32, 4, 68)
       ↓ mean over dim 0                         ↓ mean over dims 0 AND 1
Mean:  (1, 4, 68)  ← 272 means!          Mean:  (1, 1, 68)  ← 68 means ✓
       4 positions tracked separately             all positions share statistics

Each mean estimated from 32 numbers       Each mean estimated from 32×4 = 128 numbers
→ noisy estimates                         → much more stable estimates
```

### 5.2 The Fix

Use `torch.mean` with a tuple of dimensions:

```python
class BatchNorm1d:
    def __call__(self, x):
        if x.ndim == 2:
            dim = 0
        elif x.ndim == 3:
            dim = (0, 1)       # reduce over BOTH batch dimensions
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        xmean = x.mean(dim, keepdim=True)
        xvar  = x.var(dim, keepdim=True)
        # ... rest of BatchNorm as before
```

**Note on PyTorch's convention**: `torch.nn.BatchNorm1d` expects input as `(N, C)` or `(N, C, L)` — channels in the **middle**. Our implementation expects channels **last**: `(N, C)` or `(N, L, C)`. This is a deliberate departure. Karpathy prefers channels-last.

### 5.3 Impact

| Configuration | Val Loss |
| --- | --- |
| Hierarchical, BN bug (dim 0 only) | 2.029 |
| Hierarchical, BN fixed (dims 0,1) | **2.022** |

Small but real improvement — more data per statistic estimate means more stable normalization.

---

## 6. Scaling Up

With the hierarchical architecture working correctly, increase model capacity:

```python
n_embd = 24      # was 10
n_hidden = 128    # was 68
# → 76K parameters (was 22K)
```

| Configuration | Params | Val Loss |
| --- | --- | --- |
| Flat, block_size=3 | 12K | 2.10 |
| Flat, block_size=8 | 22K | 2.027 |
| Hierarchical, block_size=8 | 22K | 2.022 |
| Hierarchical, scaled up | 76K | **1.993** |

We've crossed below 2.0 for the first time. Sample quality improves noticeably.

---

## 7. Connection to Convolutions

### 7.1 What We Built Is a Convolution (Conceptually)

The hierarchical architecture processes one "tree" for a single example. But in a word like "deandre" (7 letters), there are 8 overlapping contexts. Our current code forwards each one independently:

```python
# 8 independent forward passes
for i in range(8):
    logits = model(xtr[[i]])
```

**Convolutions** let you **slide** the tree filter over the entire sequence in one efficient pass, computing all 8 outputs simultaneously.

### 7.2 Variable Reuse

```
Independent trees (our approach):      Convolution (WaveNet):

Tree for position 4:                   All trees computed at once:
  c₁ c₂ c₃ c₄                         c₁ c₂ c₃ c₄ c₅ c₆ c₇ c₈
  └─┬─┘ └─┬─┘                          └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
   bg₁₂   bg₃₄                          bg₁₂  bg₃₄  bg₅₆  bg₇₈
    └───┬───┘                             └──┬──┘ └──┬──┘
     fused                                  fg₁₄   fg₅₈
       │                                     │  ╲  ╱ │
    output₄                                  │   ╳  │
                                             │  ╱  ╲ │
Tree for position 5:                        out₄   out₅  ...
  c₂ c₃ c₄ c₅
  └─┬─┘ └─┬─┘          Shared nodes: bg₃₄ is reused!
   bg₂₃   bg₄₅         No need to recompute it.
    └───┬───┘
     fused              The "dilated causal convolution" is just
       │                this sharing done efficiently inside CUDA.
    output₅
```

Convolutions are an **implementation optimization** — the model is mathematically the same, just computed more efficiently by reusing shared intermediate computations and running the sliding in CUDA kernels.

### 7.3 What WaveNet Adds Beyond Our Model

Our implementation captures the tree structure but not the full WaveNet:

| Feature | Our model | Full WaveNet |
| --- | --- | --- |
| Hierarchical fusion | Yes | Yes |
| Dilated causal convolutions | No (explicit tree) | Yes (efficient sliding) |
| Gated activation units | No (plain tanh) | Yes (sigmoid × tanh) |
| Residual connections | No | Yes |
| Skip connections | No | Yes |

---

## 8. Development Process Notes

Karpathy shares observations about the practical deep learning workflow:

### 8.1 Shape Gymnastics

A huge amount of time is spent making tensor shapes work: is it 2D or 3D? Is it (N, C) or (N, C, L)? Promoting, viewing, squeezing. This is normal and expected.

### 8.2 Prototyping in Jupyter, Training in VS Code

Typical workflow:
1. Prototype layers and check shapes in Jupyter notebook
2. Once satisfied, copy to a proper code repository
3. Kick off training experiments from the repo

### 8.3 PyTorch Documentation

The documentation is incomplete and sometimes misleading. It prioritizes engineering features over clarity. You learn to work around it.

### 8.4 Missing: Experimental Harness

All tuning so far has been guess-and-check. A proper setup would include:
- Tracking both train and validation loss together
- Hyperparameter search scripts
- Configurable experiments with command-line arguments
- Population-level analysis of what works

---

## 9. Key Takeaways

- **Hierarchical > flat** for processing sequences: fusing information gradually through a tree preserves more signal than crushing everything into one layer
- **PyTorch `@` works on N-D tensors**: it applies matmul on the last dimension and treats everything else as batch dimensions — this is what enables the hierarchical architecture without special code
- **`view` is free**: reshaping tensors doesn't copy data, and consecutive elements naturally group correctly in memory (C-contiguous layout)
- **BatchNorm needs fixing for 3D inputs**: reduce over all batch dimensions `(0, 1)`, not just `(0)`, otherwise you maintain separate statistics per position
- **BatchNorm training mode bugs** keep appearing: forgetting to set `training = False` during evaluation is a recurring source of silent errors
- **Convolutions = efficient sliding of our tree**: the "dilated causal convolution" in WaveNet is just an efficient way to apply this tree filter at every position simultaneously
- **`torch.nn` is what we've been building**: our custom modules (Embedding, Linear, BatchNorm1d, Tanh, Sequential) mirror PyTorch's `torch.nn` — we've essentially rebuilt it from scratch

---

## 10. Looking Forward

Several directions unlocked by this lecture:
- **Convolutions**: implementing dilated causal convolution layers for efficient sliding
- **Residual and skip connections**: why they help in deeper networks
- **Experimental harness**: proper hyperparameter tuning infrastructure
- **RNNs, LSTMs, GRUs**: recurrent approaches to sequence modeling
- **Transformers**: the architecture that ultimately dominates
