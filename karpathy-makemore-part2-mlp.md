# makemore Part 2: MLP Language Model

**Source**: Andrej Karpathy's "makemore" series, Part 2
**Paper**: Bengio et al. (2003) — *A Neural Probabilistic Language Model*
**Prerequisites**: Bigram model from Part 1, basic PyTorch

---

## Overview

This lecture moves from the simple bigram character-level language model (Part 1) to a **multi-layer perceptron (MLP)** that uses multiple characters of context to predict the next character. The architecture follows Bengio et al. (2003), which embeds input tokens into a continuous vector space, concatenates them, passes them through a hidden layer, and produces a probability distribution over the next token. The lecture covers the full pipeline: dataset construction, model architecture, training with mini-batches, learning rate tuning, overfitting detection, and sampling.

---

## 1. Why Move Beyond Bigrams?

The bigram model only looks at **1 previous character** to predict the next one. This means:

- 27 possible contexts (26 letters + 1 special character)
- Predictions are poor because there's almost no context

If we try to scale the count-based approach to more context, the table **grows exponentially**:

| Context length | Number of rows |
|---------------|---------------|
| 1 character   | 27            |
| 2 characters  | 27 x 27 = 729 |
| 3 characters  | 27^3 = 19,683 |
| 4 characters  | 27^4 = 531,441 |

With more rows, each row has fewer training examples, counts become sparse, and the model can't generalize. Neural networks solve this by learning **distributed representations** rather than maintaining explicit count tables.

---

## 2. The MLP Architecture (Bengio et al. 2003)

### 2.1 High-Level Flow

```
Input characters (integers)
        |
        v
Embedding lookup table C    (each character -> dense vector)
        |
        v
Concatenate embeddings      (flatten into one long vector)
        |
        v
Hidden layer + tanh          (learned nonlinear transformation)
        |
        v
Output layer                 (logits for each possible next character)
        |
        v
Softmax                      (convert logits to probabilities)
        |
        v
Cross-entropy loss           (compare prediction to actual next character)
```

### 2.2 Embedding Lookup Table

Every character is mapped to a learned vector. Instead of treating characters as isolated symbols, the network places them in a **continuous space** where similar characters can end up near each other.

```python
# 27 characters, each embedded into a 10-dimensional space
C = torch.randn((27, 10))

# Embed all inputs at once — X is shape (N, block_size) of integers
emb = C[X]  # shape: (N, block_size, 10)
```

```
C (embedding table):            X (input):        emb = C[X]:
  27 rows x 10 cols              N x 3             N x 3 x 10

  char  dim0  dim1 ... dim9     [0, 0, 5]        [C[0], C[0], C[5]]
  ---- ----- ----- --- -----    [0, 5, 13]  -->  [C[0], C[5], C[13]]
  '.'  [0.2, -0.1, ... 0.5]    [5, 13, 13]       [C[5], C[13], C[13]]
  'a'  [0.8,  0.3, ... -0.2]    ...                ...
  'b'  [-0.1, 0.7, ... 0.4]
   :      :     :        :     Each integer in X
  'z'  [0.3, -0.5, ... 0.1]    plucks out a row from C
```

**Why this works (intuition)**: If the network learns that "a" and "the" appear in similar contexts, their embeddings will be pushed close together during training. This lets the network generalize — even if it never saw "a dog was running in a ___", it can transfer knowledge from "the dog was running in a ___" because the embeddings for "a" and "the" are nearby.

### 2.3 Embedding as a Neural Network Layer

Indexing `C[5]` is mathematically equivalent to multiplying a one-hot vector by `C`:

```python
# These two are identical:
C[5]                                          # direct lookup
F.one_hot(torch.tensor(5), num_classes=27).float() @ C  # matrix multiply
```

```
One-hot for index 5:                C (27 x 10):           Result:

[0, 0, 0, 0, 0, 1, 0, ..., 0]  @  [row 0: 0.2, -0.1, ...]   = [row 5 of C]
         (1 x 27)                   [row 1: 0.8,  0.3, ...]     = [-0.3, 0.9, ...]
                                    [ ...                  ]       (1 x 10)
                                    [row 5: -0.3, 0.9, ...]
                                    [ ...                  ]
                                    [row 26: 0.3, -0.5, ...]

The 1 at position 5 "selects" row 5 — everything else is zeroed out.
C[5] does the same thing instantly, without creating the one-hot vector.
```

The one-hot interpretation shows that the embedding table is just **the weight matrix of a linear layer with no bias and no activation**. We use direct indexing because it's much faster.

### 2.4 Full Model Definition

```python
import torch
import torch.nn.functional as F

block_size = 3   # context length: how many characters predict the next one
n_embd = 10      # embedding dimensions
n_hidden = 200   # neurons in hidden layer
vocab_size = 27   # number of possible characters

g = torch.Generator().manual_seed(2147483647)

C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g)
W2 = torch.randn((n_hidden, vocab_size),           generator=g)
b2 = torch.randn(vocab_size,                       generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

print(f"Total parameters: {sum(p.nelement() for p in parameters)}")
```

```
Parameter shapes:

C:   (27, 10)    — 27 characters, each a 10-dim embedding vector
W1:  (30, 200)   — 30 inputs (3 chars × 10 dims) → 200 hidden neurons
b1:  (200,)      — one bias per hidden neuron
W2:  (200, 27)   — 200 hidden → 27 output logits (one per character)
b2:  (27,)       — one bias per output character

Total: 27×10 + 30×200 + 200 + 200×27 + 27 = 11,697 parameters
```

### 2.5 Forward Pass

```python
# 1. Embed the input characters
emb = C[X]                            # (N, block_size, n_embd)

# 2. Concatenate embeddings by reshaping
h_preact = emb.view(-1, n_embd * block_size) @ W1 + b1  # (N, n_hidden)

# 3. Apply nonlinearity
h = torch.tanh(h_preact)              # (N, n_hidden)

# 4. Output layer
logits = h @ W2 + b2                  # (N, vocab_size)

# 5. Loss
loss = F.cross_entropy(logits, Y)
```

```
Step-by-step shape transformations (N = batch size, e.g., 32):

Step 1: Embed
  X:   (32, 3)          — 32 examples, each 3 character indices
  emb: (32, 3, 10)      — each index replaced by its 10-dim embedding

Step 2: Reshape + Linear
  emb.view: (32, 30)    — flatten 3×10 into 30 (concatenate embeddings)

  (32, 30) @ (30, 200) + (200,) = (32, 200)
   ──emb──    ───W1───   ──b1──   ─h_preact─

Step 3: Nonlinearity
  h = tanh(h_preact):  (32, 200)   — squash each value to [-1, +1]

Step 4: Output
  (32, 200) @ (200, 27) + (27,) = (32, 27)
   ───h────    ───W2───   ──b2──  ─logits─

Step 5: Loss
  logits: (32, 27)  — one score per character per example
  Y:      (32,)     — correct character index for each example
  loss:   scalar    — average cross-entropy over the batch
```

---

## 3. Key Implementation Details

### 3.1 Dataset Construction

The dataset is built using a **sliding window** of size `block_size`. Each word is padded with special "." characters (index 0) at the start.

```python
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # start with padding
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide window forward
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
```

For the word "emma", this generates:

```
Context (X)        Label (Y)      As indices:
-----------        ---------      -----------
 . . .      -->      e            [0,  0,  0] --> 5
 . . e      -->      m            [0,  0,  5] --> 13
 . e m      -->      m            [0,  5, 13] --> 13
 e m m      -->      a            [5, 13, 13] --> 1
 m m a      -->      .            [13, 13, 1] --> 0
```

### 3.2 PyTorch Indexing Power

PyTorch indexing works with scalars, lists, 1D tensors, and **multi-dimensional tensors**:

```python
C[5]              # single row — shape (n_embd,)
C[[5, 6, 7]]      # three rows — shape (3, n_embd)
C[X]              # X is (N, block_size) — result is (N, block_size, n_embd)
```

```
C[5]           → [0.3, -0.1, ..., 0.8]              shape: (10,)

C[[5, 6, 7]]   → [[0.3, -0.1, ..., 0.8],            shape: (3, 10)
                   [0.1,  0.4, ..., -0.2],
                   [-0.5, 0.2, ..., 0.6]]

C[X]           X = [[0,  0,  5],                     shape: (4, 3)
                    [0,  5, 13],
                    [5, 13, 13],
                    [13, 13, 1]]

               C[X] = [[[C[0]], [C[0]], [C[5]]],     shape: (4, 3, 10)
                        [[C[0]], [C[5]], [C[13]]],    each C[i] is a
                        [[C[5]], [C[13]],[C[13]]],    10-dim vector
                        [[C[13]],[C[13]],[C[1]]]]
```

This is what makes `C[X]` so elegant — it embeds the entire batch in one operation.

### 3.3 `tensor.view()` vs `torch.cat()` — Reshaping Without Copying

To concatenate the embeddings of 3 characters into a single vector, there are two approaches:

**Approach 1: `torch.cat` (allocates new memory)**

```python
# Works but creates a new tensor in memory
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)  # (N, 30)
# Or more generally:
torch.cat(torch.unbind(emb, dim=1), dim=1)  # (N, 30)
```

**Approach 2: `tensor.view()` (no memory copy)**

```python
# Just reinterprets the same memory — much faster
emb.view(-1, n_embd * block_size)  # (N, 30)
```

```
emb shape: (32, 3, 10)

One example row (3 chars, each 10-dim):
 char 0 embedding   char 1 embedding   char 2 embedding
 [0.2, -0.1, ...]   [0.8, 0.3, ...]   [-0.3, 0.9, ...]
   (10 values)         (10 values)        (10 values)

After .view(-1, 30) — same example becomes one flat row:
 [0.2, -0.1, ..., 0.8, 0.3, ..., -0.3, 0.9, ...]
              (30 values total)

Result shape: (32, 30)  — same data, just reinterpreted
```

**Why `.view()` is better**: PyTorch tensors have an underlying 1D storage in memory. `.view()` only changes the metadata (shape, strides) — it doesn't move or copy any data. `.cat()` must allocate entirely new memory and copy values into it.

The `-1` in `.view(-1, 30)` tells PyTorch to infer that dimension automatically.

### 3.4 Why Use `F.cross_entropy` Instead of Manual Softmax

Manual implementation:

```python
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
loss = -probs[torch.arange(N), Y].log().mean()
```

```
logits (32, 27):     exp()→ counts:      normalize → probs:
                                          (each row sums to 1)
[ 2.1, -0.3, ...]   [8.2, 0.7, ...]     [0.15, 0.01, ...]  ← example 0
[-1.0,  3.5, ...]   [0.4, 33.1, ...]    [0.00, 0.62, ...]  ← example 1
  ...                  ...                 ...

If Y = [5, 13, ...]:
  probs[0, 5]  = 0.03  ← prob assigned to correct char for example 0
  probs[1, 13] = 0.07  ← prob assigned to correct char for example 1
  loss = -mean(log(0.03), log(0.07), ...)
```

This works but has three problems that `F.cross_entropy` solves:

**1. Efficiency — fused kernels**
PyTorch fuses the exponentiation, normalization, log, and indexing into a single optimized kernel instead of creating multiple intermediate tensors.

**2. Simpler backward pass**
The combined derivative of cross-entropy + softmax simplifies analytically to `(p - y)`, which is much cheaper to compute than backpropagating through each operation individually. (Same principle as how tanh backward simplifies to `1 - t^2` instead of backpropagating through the full expression.)

**3. Numerical stability**
Large positive logits cause `exp()` to overflow to infinity:

```python
logits = torch.tensor([100.0, 0.0, 0.0])
logits.exp()  # tensor([inf, 1., 1.]) — NaN will follow

# PyTorch internally subtracts the max first:
# softmax(x) == softmax(x - max(x))  — mathematically provable
(logits - logits.max()).exp()  # safe, no overflow
```

Negative logits are fine (`exp(-100)` is just a very small number near zero), but positive ones can overflow. `F.cross_entropy` handles this automatically.

---

## 4. Training Practices

### 4.1 Mini-Batch Stochastic Gradient Descent

With 228,000 examples, computing the gradient over the full dataset each step is extremely slow. Instead, sample a **random mini-batch** each iteration:

```python
batch_size = 32
ix = torch.randint(0, X_train.shape[0], (batch_size,))

# Forward pass on just the mini-batch
emb = C[X_train[ix]]
# ... rest of forward pass using only batch_size examples ...
loss = F.cross_entropy(logits, Y_train[ix])
```

```
Full dataset:                    Mini-batch (ix = [7, 102, 55841, ...]):
X_train: (228000, 3)            X_train[ix]: (32, 3)
Y_train: (228000,)     ──→     Y_train[ix]: (32,)
                         random
                         sample    Only 32 examples!
                                   ~7000x less work per step
```

**Why this works**: The gradient computed on 32 random examples is noisy but directionally useful. Making **many fast approximate steps** beats making **few slow exact steps**. This is the fundamental insight behind SGD.

### 4.2 Learning Rate Finding

Don't guess the learning rate. Sweep it systematically:

```python
# Create 1000 learning rates, spaced exponentially from 10^-3 to 10^0
lre = torch.linspace(-3, 0, 1000)  # exponents
lrs = 10 ** lre                     # actual learning rates

lri, lossi = [], []
for i in range(1000):
    # ... forward pass ...
    # ... backward pass ...

    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad

    lri.append(lre[i])
    lossi.append(loss.item())

# Plot: x-axis = learning rate exponent, y-axis = loss
plt.plot(lri, lossi)
```

The resulting plot typically looks like:

```
Loss
 |
 |  ----___
 |         \___         <-- pick somewhere in this valley
 |             \___
 |                 \
 |                  |
 |                  /  <-- loss explodes (LR too high)
 |                 /
 +-------------------> LR exponent
   -3   -2   -1    0
```

Pick the learning rate in the **valley** — not too slow (left), not exploding (right). In this lecture, the sweet spot was around `10^-1 = 0.1`.

### 4.3 Learning Rate Decay

After training at a good learning rate for a while, **reduce it** (e.g., by 10x) to squeeze out the last performance gains:

```python
for i in range(200000):
    # ... forward + backward ...

    lr = 0.1 if i < 100000 else 0.01  # decay at halfway point
    for p in parameters:
        p.data += -lr * p.grad
```

This is a simple step decay. The initial rate makes fast progress; the reduced rate fine-tunes without overshooting.

### 4.4 Identifying Bottlenecks

When the model isn't improving, the **constraint might not be where you think**:

| What was tried | Result | Conclusion |
|---|---|---|
| Hidden layer 100 -> 300 neurons | Loss barely improved | Hidden layer wasn't the bottleneck |
| Embedding dim 2 -> 10 | Loss improved significantly | 2D embeddings were too small to represent character relationships |

**Lesson**: Scale one component, observe. If no improvement, the bottleneck is elsewhere. Try scaling a different component.

---

## 5. Overfitting & Data Splits

### 5.1 Three-Way Split

| Split | Proportion | Purpose |
|-------|-----------|---------|
| **Training** | 80% (~25,000 words) | Optimize parameters via gradient descent |
| **Dev / Validation** | 10% (~3,000 words) | Tune hyperparameters, monitor generalization |
| **Test** | 10% (~3,200 words) | Final evaluation — use **very sparingly** |

```python
import random
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

X_train, Y_train = build_dataset(words[:n1])
X_dev,   Y_dev   = build_dataset(words[n1:n2])
X_test,  Y_test  = build_dataset(words[n2:])
```

### 5.2 Diagnosing Under/Overfitting

| Signal | Diagnosis | Action |
|--------|-----------|--------|
| Train loss ~ Dev loss (both high) | **Underfitting** | Make model bigger, train longer |
| Train loss << Dev loss | **Overfitting** | Model is memorizing; add regularization, reduce size, get more data |
| Train loss ~ Dev loss (both low) | **Good fit** | Model generalizes well |

### 5.3 Why Not Just Use One Split?

- If you tune hyperparameters on the test set, you're **indirectly training on it** — each decision you make based on test performance is a form of fitting.
- The dev set absorbs this. The test set stays untouched until the very end.
- Every time you look at test loss and adjust something, you risk overfitting to it.

---

## 6. Concepts Clarified

### 6.1 Parameters vs Hyperparameters

| | Parameters | Hyperparameters |
|---|---|---|
| **What** | Values inside the model | Design choices for the model |
| **Learned by** | Gradient descent (automatic) | You (manual experiments) |
| **Evaluated on** | Training set | Dev/validation set |
| **Examples** | `W1`, `W2`, `b1`, `b2`, `C` | Hidden layer size, embedding dim, learning rate, block size, batch size |
| **Count in this model** | ~11,000 | ~6 choices |

### 6.2 Embedding Space Visualization

When the embedding dimension is 2, you can plot all characters on a 2D plane. After training:

- **Vowels** (a, e, i, o, u) **cluster together** — the network learned they appear in similar contexts
- **Special characters** (`.`, `q`) are **far away** — they behave very differently from typical letters
- Other consonants form their own loose grouping

This is evidence that the network learns meaningful structure, not random noise.

### 6.3 Empirical vs Theoretically Grounded Tips

| Tip | Basis |
|-----|-------|
| Learning rate sweep | **Empirical** — no formula gives optimal LR |
| Learning rate decay schedule | **Empirical** — when/how much is trial and error |
| Mini-batch size choice | **Empirical** — 32 vs 64 vs 128 is experimentation |
| Bottleneck identification | **Empirical** — scale parts, observe, hypothesize |
| `F.cross_entropy` numerical stability | **Theoretical** — `softmax(x) = softmax(x - c)` is provable |
| `.view()` vs `.cat()` efficiency | **Theoretical** — follows from tensor memory layout |
| Train/dev/test splits | **Theoretical** — rooted in statistical learning theory |
| Cross-entropy backward simplifies to `(p - y)` | **Theoretical** — analytically derivable |
| SGD convergence with noisy gradients | **Both** — theory proves convergence, practice determines optimal batch size |

---

## 7. Key Takeaways

- **Embeddings are powerful**: mapping discrete tokens to continuous vectors enables generalization through similarity.
- **`tensor.view()` is free**: it doesn't copy memory. Always prefer it over `torch.cat` when possible.
- **Always use `F.cross_entropy`**: it's faster, numerically stable, and has a simpler backward pass.
- **Mini-batching is essential**: noisy but fast steps beat exact but slow steps.
- **Find your learning rate systematically**: sweep exponentially, plot loss, pick the valley.
- **Diagnose bottlenecks by ablation**: change one thing at a time, observe what helps.
- **Split your data three ways**: train on training, tune on dev, report on test.
- **The best validation loss achieved in the lecture**: **2.17** (vs 2.45 for bigrams from Part 1).

### Training Checklist

- [ ] Set up data splits (train / dev / test)
- [ ] Initialize model with reasonable hyperparameters
- [ ] Run learning rate sweep to find good starting LR
- [ ] Train with mini-batches, monitor loss
- [ ] Compare train loss vs dev loss to diagnose under/overfitting
- [ ] If underfitting: increase model size (embeddings, hidden layer, context length)
- [ ] If overfitting: reduce model size, add regularization, get more data
- [ ] Apply learning rate decay when loss plateaus
- [ ] Report final test loss only once

---

## 8. Exercises

From the lecture, try to beat the **2.17 validation loss** by tuning:

1. **Embedding dimension** — try 2, 10, 20, 50
2. **Hidden layer size** — try 100, 200, 300, 500
3. **Context length (block_size)** — try 3, 5, 8
4. **Learning rate schedule** — try different initial LR, different decay points
5. **Batch size** — try 32, 64, 128, 256
6. **Training duration** — how many steps before diminishing returns?

Additionally, read the full Bengio et al. (2003) paper — at this point most of it should be accessible.

---

*Notes based on Andrej Karpathy's makemore lecture series, Part 2.*
