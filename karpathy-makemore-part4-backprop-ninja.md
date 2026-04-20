# makemore Part 4: Becoming a Backprop Ninja

**Source**: Andrej Karpathy's "makemore" series, Part 4
**Prerequisites**: MLP with batch normalization from Part 3, micrograd from Part 1
**Theme**: Manual backpropagation through tensors — understanding what `loss.backward()` actually does

---

## Overview

This lecture pauses before moving to RNNs to build a deep, hands-on understanding of **backpropagation at the tensor level**. Instead of relying on PyTorch's autograd (`loss.backward()`), we manually derive and implement the backward pass for every operation in our two-layer MLP with batch normalization. The lecture covers four exercises: (1) element-by-element backpropagation through every atomic operation, (2) analytically deriving the efficient cross-entropy backward pass, (3) analytically deriving the efficient batch normalization backward pass, and (4) putting it all together to train without autograd.

**Key argument**: Backpropagation is a **leaky abstraction**. Even though frameworks compute gradients automatically, not understanding the internals leads to subtle bugs — dead neurons, vanishing/exploding gradients, accidentally zeroing out outlier gradients by clipping loss instead of clipping gradients, etc.

---

## 1. Why Manual Backpropagation?

### 1.1 Backprop as a Leaky Abstraction

You cannot just stack arbitrary differentiable Lego blocks and hope `loss.backward()` makes everything work. Common pitfalls that require understanding backprop:

- **Saturated activations** (tanh/sigmoid flat tails → gradient dies)
- **Dead ReLU neurons** (permanently zero gradient)
- **Exploding/vanishing gradients** in deep/recurrent networks
- **Subtle bugs**: e.g., clipping the *loss* at a maximum value when you meant to clip the *gradient* — this silently zeroes out gradients for outlier examples instead of capping them

### 1.2 Historical Context

Manual backward passes were standard practice until ~2015:


| Era                            | Practice                                                          |
| ------------------------------ | ----------------------------------------------------------------- |
| ~2006 (Hinton & Salakhutdinov) | Hand-written backward in Matlab for restricted Boltzmann machines |
| ~2010 (Karpathy's own code)    | Matlab-based RBM training, gradients computed inline              |
| ~2014 (Karpathy's papers)      | NumPy-based, cost function + backward pass + gradient checker     |
| ~2016+                         | Autograd frameworks (Theano, TensorFlow, PyTorch) take over       |


The key difference: in micrograd we backpropagated through **scalars**. Now we backpropagate through **tensors** — the same chain rule, but with matrix calculus and broadcasting.

---

## 2. Setup: The Forward Pass (Expanded)

The network is the same two-layer MLP with batch normalization from Part 3, but the forward pass is broken into atomic operations so we can backpropagate through each one:

```python
# ---- Forward Pass (expanded) ----

# Embedding
emb = C[Xb]                                     # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)             # (32, 30)

# Layer 1
hprebn = embcat @ W1 + b1                        # (32, 64)

# BatchNorm
bnmeani = hprebn.sum(0, keepdim=True) / n         # (1, 64)
bndiff = hprebn - bnmeani                         # (32, 64)
bndiff2 = bndiff ** 2                             # (32, 64)
bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)   # (1, 64)  ← Bessel's correction
bnvar_inv = (bnvar + 1e-5) ** (-0.5)             # (1, 64)
bnraw = bndiff * bnvar_inv                        # (32, 64)
hpreact = bngain * bnraw + bnbias                 # (32, 64)

# Activation
h = torch.tanh(hpreact)                           # (32, 64)

# Layer 2
logits = h @ W2 + b2                              # (32, 27)

# Loss (explicit softmax + cross-entropy)
logit_maxes = logits.max(1, keepdim=True).values  # (32, 1)
norm_logits = logits - logit_maxes                # (32, 27)
counts = norm_logits.exp()                        # (32, 27)
counts_sum = counts.sum(1, keepdim=True)          # (32, 1)
counts_sum_inv = counts_sum ** (-1)               # (32, 1)
probs = counts * counts_sum_inv                   # (32, 27)
logprobs = probs.log()                            # (32, 27)
loss = -logprobs[range(n), Yb].mean()             # scalar
```

```
Forward pass data flow (with shapes):

C (27, 10)   Xb (32, 3)
      \       /
    emb (32, 3, 10)
         |  .view()
    embcat (32, 30)
         |          W1 (30, 64)   b1 (64,)
         +----@------+-----+------+
              |
         hprebn (32, 64)
              |
     ┌── bnmeani (1, 64) ──┐
     |        |              |
     |   bndiff (32, 64)     |
     |     |       |         |
     |   bndiff2  bnvar (1,64)
     |     |       |
     |     |   bnvar_inv (1, 64)
     |     |       |
     |     └─bnraw (32, 64)──┘
     |         |
     |    bngain (1, 64)  bnbias (1, 64)
     |         |              |
     └──→ hpreact (32, 64)←──┘
              |  tanh
         h (32, 64)
              |          W2 (64, 27)  b2 (27,)
              +----@------+-----+-----+
                   |
            logits (32, 27)
                   |
           logit_maxes (32, 1)    ← max per row
                   |
          norm_logits (32, 27)    ← subtract max
                   |  exp()
            counts (32, 27)
                   |
          counts_sum (32, 1)      ← sum per row
                   |  ** (-1)
        counts_sum_inv (32, 1)
                   |
            probs (32, 27)        ← normalized
                   |  log()
          logprobs (32, 27)
                   |  index + mean
              loss (scalar)
```

**Note on initialization**: biases are initialized to small random numbers (not zero) to avoid masking bugs. `b1` is kept despite BatchNorm to practice computing its gradient.

---

## 3. Exercise 1: Element-by-Element Backpropagation

### 3.1 Gradient Checking Setup

A utility function compares our manual gradients against PyTorch's autograd:

```python
def cmp(s, dt, t):
    ex  = torch.all(dt == t.grad).item()          # exact match?
    app = torch.allclose(dt, t.grad)               # approximate match?
    maxdiff = (dt - t.grad).abs().max().item()     # largest difference
    print(f'{s:15s} | exact: {str(ex):5s} | approx: {str(app):5s} | maxdiff: {maxdiff}')
```

### 3.2 dlogprobs — Backprop Through the Mean

The loss is:

```
loss = -( logprobs[0,y0] + logprobs[1,y1] + ... + logprobs[31,y31] ) / n
```

Only 32 elements of the (32, 27) `logprobs` tensor participate. For each participating element, the derivative is `-1/n`. All others are zero.

```python
dlogprobs = torch.zeros_like(logprobs)         # (32, 27) of zeros
dlogprobs[range(n), Yb] = -1.0 / n            # place -1/n at correct positions
```

```
logprobs (32 × 27):                    dlogprobs (32 × 27):

      col 0   col 1  ...  col y0       col 0   col 1  ...  col y0
row 0 [-3.2,  -4.1, ...,  -2.8 ]  →   [ 0,     0,    ..., -1/32 ]
row 1 [-3.5,  -2.9, ...,  -3.1 ]  →   [ 0,     0,    ...,  0    ]
 ...                  col y1                             col y1
                     [-2.5 ]                            [-1/32 ]
                       ↑                                   ↑
                   participates                      gets gradient
                   in loss                           (-1/n)

All other entries: derivative = 0 (don't affect loss)
```

**Concrete numerical walkthrough** (3 examples, 4 classes for simplicity):

```
Suppose n=3, labels Yb = [2, 0, 3]

logprobs (3 × 4):
         class 0   class 1   class 2   class 3
ex 0:  [ -1.8,     -2.1,     -0.9,     -3.2  ]  ← label=2
ex 1:  [ -1.1,     -2.5,     -1.6,     -2.0  ]  ← label=0
ex 2:  [ -3.0,     -1.4,     -2.2,     -0.8  ]  ← label=3

Step 1: pluck out correct entries
  logprobs[0, 2] = -0.9
  logprobs[1, 0] = -1.1
  logprobs[2, 3] = -0.8

Step 2: loss = -(-0.9 + -1.1 + -0.8) / 3 = -(−2.8) / 3 = 0.933

Step 3: dlogprobs — who contributed and how?
  dlogprobs (3 × 4):
         class 0    class 1    class 2    class 3
ex 0:  [  0,         0,       -1/3,        0    ]  ← only [0,2]
ex 1:  [ -1/3,       0,         0,         0    ]  ← only [1,0]
ex 2:  [  0,         0,         0,       -1/3   ]  ← only [2,3]

The -1/3 comes from: loss = -(a + b + c)/3
  so ∂loss/∂a = -1/3  (same for b and c)

All the zeros: those logprobs never appeared in the loss,
  so changing them changes nothing. Gradient = 0.
```

### 3.3 dprobs — Backprop Through log

`logprobs = log(probs)`, applied element-wise. The derivative of log(x) is 1/x.

```python
dprobs = (1.0 / probs) * dlogprobs              # (32, 27)
```

**Intuition**: if a probability is very small (model is wrong), `1/probs` is very large, amplifying the gradient signal — the model gets a strong push to correct its mistake.

```
Numerical example — how 1/probs amplifies wrong predictions:

Suppose for two different examples, the correct class has:

Example A: probs[correct] = 0.80  (model is mostly right)
  1/probs = 1/0.80 = 1.25         ← mild amplification
  dlogprobs = -1/32 = -0.03125
  dprobs = 1.25 × (-0.03125) = -0.039

Example B: probs[correct] = 0.02  (model is badly wrong)
  1/probs = 1/0.02 = 50.0         ← 40× stronger amplification!
  dlogprobs = -1/32 = -0.03125
  dprobs = 50.0 × (-0.03125) = -1.5625

The worse the prediction, the stronger the gradient signal.
The model gets screamed at for confident wrong predictions,
but only gets a gentle nudge when it's already doing well.
```

### 3.4 dcounts_sum_inv — Backprop Through Element-wise Multiply with Broadcasting

`probs = counts * counts_sum_inv` where counts is (32, 27) and counts_sum_inv is (32, 1).

```
Forward: broadcasting in element-wise multiply

counts (32 × 27):         counts_sum_inv (32 × 1):       probs (32 × 27):
[c00, c01, ..., c0,26]      [s0]  → replicated →         [c00·s0, c01·s0, ...]
[c10, c11, ..., c1,26]  ×   [s1]  → [s0,s0,...,s0]   =   [c10·s1, c11·s1, ...]
 ...                         ...     [s1,s1,...,s1]         ...
                                      ...
```

For `counts_sum_inv`: local derivative is `counts`, then chain rule × `dprobs`. But `counts_sum_inv` was broadcast (replicated across columns), so we **sum** to undo the replication:

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)  # (32, 1)
```

```
Backward: sum undoes the broadcasting

dprobs (32 × 27):                          dcounts_sum_inv (32 × 1):
[dp00, dp01, ..., dp0,26]                    [Σ(counts[0,:] * dprobs[0,:])]
[dp10, dp11, ..., dp1,26]  → sum each row →  [Σ(counts[1,:] * dprobs[1,:])]
 ...                                           ...
```

**Key duality**: forward broadcasting (replication) ↔ backward summation, and vice versa.

**Concrete example** (2 examples, 3 classes):

```
Forward:
counts:          counts_sum_inv:       probs:
[2, 5, 3]         [1/10]        →    [2/10, 5/10, 3/10] = [0.2, 0.5, 0.3]
[1, 1, 8]    ×    [1/10]        →    [1/10, 1/10, 8/10] = [0.1, 0.1, 0.8]
                 (broadcast)

Backward (suppose dprobs is):
dprobs:
[-0.5,  0.1,  0.0]
[ 0.0,  0.0, -0.3]

dcounts_sum_inv = sum of (counts * dprobs) per row:
  row 0: 2×(-0.5) + 5×(0.1) + 3×(0.0) = -1.0 + 0.5 + 0 = -0.5
  row 1: 1×(0.0) + 1×(0.0) + 8×(-0.3) = 0 + 0 + (-2.4) = -2.4
  result: [-0.5]    ← the sum "collapsed" 3 columns into 1
          [-2.4]
```

For `counts` (the first branch — there will be a second branch later):

```python
dcounts = counts_sum_inv * dprobs              # (32, 27)
# counts_sum_inv (32,1) broadcasts to match dprobs (32,27)
```

### 3.5 dcounts_sum — Backprop Through Power of -1

`counts_sum_inv = counts_sum ** (-1)`. Derivative of x^(-1) is -x^(-2):

```python
dcounts_sum = (-counts_sum ** (-2)) * dcounts_sum_inv   # (32, 1)
```

**Intuition**: when `counts_sum` is large, the inverse `1/counts_sum` is small, and its derivative `-1/counts_sum²` is even smaller. A large normalizing constant is "stable" — small changes to it barely affect the output.

```
Numerical example:

counts_sum = 100     → counts_sum_inv = 0.01
  derivative = -1/100² = -0.0001    ← tiny! stable normalization

counts_sum = 2       → counts_sum_inv = 0.5
  derivative = -1/2² = -0.25        ← much larger! sensitive to changes

When probabilities are spread across many high-count classes,
the normalization denominator is large and gradients are small.
When only a few classes have counts, the denominator is small
and gradients are larger.
```

### 3.6 dcounts (second branch) — Backprop Through Sum

`counts_sum = counts.sum(1, keepdim=True)` sums each row. In the backward pass, the gradient on the sum just gets **routed equally** to all elements that participated.

```python
dcounts += torch.ones_like(counts) * dcounts_sum  # (32, 27)
# dcounts_sum (32,1) broadcasts, adding its gradient to all 27 positions per row
# += because this is the SECOND branch contributing to dcounts
```

```
Forward: sum collapses columns          Backward: gradient replicates

counts (32 × 27):                       dcounts (32 × 27):
[c00, c01, ..., c0,26]                  [ds0, ds0, ..., ds0]  ← same value
      ↓  sum each row                         ↑  replicate
counts_sum (32 × 1):                    dcounts_sum (32 × 1):
[Σ row 0]                               [ds0]
[Σ row 1]                               [ds1]
```

**Intuition**: addition is a **gradient router**. When you compute `b = a1 + a2 + a3`, all three inputs contributed equally to the output. If someone tells you "b needs to increase by 0.5" (i.e., db = 0.5), that message gets forwarded identically to all three inputs: da1 = da2 = da3 = 0.5. The sum doesn't amplify or diminish — it just copies the gradient to every contributor.

### 3.7 dnorm_logits — Backprop Through exp

`counts = exp(norm_logits)`. The derivative of e^x is e^x, which we already computed as `counts`:

```python
dnorm_logits = counts * dcounts                  # (32, 27)
```

**Intuition**: `exp()` amplifies gradients for large logits and dampens them for small/negative logits. This is because the local derivative of e^x is e^x itself.

```
Numerical example:

norm_logits:  [ 2.0,    0.0,   -3.0  ]
counts=exp(): [ 7.39,   1.0,    0.05 ]  ← these ARE the local derivatives
dcounts:      [ 0.1,    0.1,    0.1  ]  ← suppose equal incoming gradients

dnorm_logits: [7.39×0.1, 1.0×0.1, 0.05×0.1]
            = [ 0.739,   0.1,     0.005 ]

The logit at 2.0 gets a gradient 148× larger than the logit at -3.0!
This makes sense: exp(2) is a steep part of the curve (sensitive),
while exp(-3) is nearly flat (insensitive).
```

### 3.8 dlogits and dlogit_maxes — Backprop Through Subtraction with Broadcasting

`norm_logits = logits - logit_maxes` where logits is (32, 27) and logit_maxes is (32, 1).

```python
dlogits = dnorm_logits.clone()                   # (32, 27) — gradient passes through
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)  # (32, 1) — negative, then sum
```

The negative sign comes from the subtraction (`-logit_maxes`). The sum comes from undoing the broadcasting.

**Important insight**: `logit_maxes` exists only for numerical stability. Changing it doesn't change the softmax output or the loss. Therefore `dlogit_maxes` should be approximately **zero** — and indeed, inspecting it shows values on the order of 1e-9.

### 3.9 dlogits (second branch) — Backprop Through max

`logit_maxes = logits.max(1, keepdim=True)`. The max operation plucks out one element per row. The gradient flows only to that element:

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=27).float() * dlogit_maxes
# one_hot creates a mask: 1 at max position, 0 elsewhere
# dlogit_maxes (32,1) broadcasts to (32,27), then masked
```

```
logits row 0: [0.2, 1.5, -0.3, ..., 0.8]
                     ↑ max at index 1

one_hot:      [0,   1,    0,   ...,  0  ]  ← only max position gets gradient

dlogit_maxes: [≈0]  (numerically tiny — max doesn't affect loss)
```

### 3.10 dh, dW2, db2 — Backprop Through Linear Layer

`logits = h @ W2 + b2` where h is (32, 64), W2 is (64, 27), b2 is (27,).

**Deriving matrix multiplication backward from first principles**:

For a small example, write out `D = A @ B + C` element by element:

```
A (2×2)       B (2×2)       C (1×2, broadcast)     D (2×2)
[a11, a12]    [b11, b12]    [c1, c2]               [d11, d12]
[a21, a22]  @ [b21, b22]  + [c1, c2]            =  [d21, d22]

Expanded:
d11 = a11·b11 + a12·b21 + c1
d12 = a11·b12 + a12·b22 + c2
d21 = a21·b11 + a22·b21 + c1
d22 = a21·b12 + a22·b22 + c2
```

Now differentiate. For example, `dL/da11`:

- `a11` appears in `d11` (with coefficient `b11`) and `d12` (with coefficient `b12`)
- `dL/da11 = dL/dd11 · b11 + dL/dd12 · b12`

Writing out all elements of `dL/dA`, you discover it's a matrix multiplication:

```
dL/dA = dL/dD @ B^T

[dL/da11, dL/da12]   [dL/dd11, dL/dd12]   [b11, b21]
[dL/da21, dL/da22] = [dL/dd21, dL/dd22] @ [b12, b22]
                         dL/dD                B^T
```

Similarly: `dL/dB = A^T @ dL/dD` and `dL/dC = sum(dL/dD, axis=0)`.

**The shape-matching trick** — you don't need to memorize these formulas:

```
Want dh:  shape must be (32, 64)
Have: dlogits (32, 27), W2 (64, 27)
Only way: dlogits @ W2.T  →  (32,27) @ (27,64) = (32,64) ✓

Want dW2: shape must be (64, 27)
Have: h (32, 64), dlogits (32, 27)
Only way: h.T @ dlogits  →  (64,32) @ (32,27) = (64,27) ✓

Want db2: shape must be (27,)
Have: dlogits (32, 27)
Only way: dlogits.sum(0)  →  (27,) ✓
```

```python
dh = dlogits @ W2.T                             # (32, 64)
dW2 = h.T @ dlogits                             # (64, 27)
db2 = dlogits.sum(0)                            # (27,)
```

**Full numerical walkthrough** (2×2 matrices):

```
Forward:
A = [[1, 2],     B = [[5, 6],     C = [0.1, 0.2]  (broadcast)
     [3, 4]]          [7, 8]]

D = A @ B + C:
  d11 = 1×5 + 2×7 + 0.1 = 19.1     d12 = 1×6 + 2×8 + 0.2 = 22.2
  d21 = 3×5 + 4×7 + 0.1 = 43.1     d22 = 3×6 + 4×8 + 0.2 = 50.2

D = [[19.1, 22.2],
     [43.1, 50.2]]

Suppose dL/dD = [[1.0, 0.5],     (given from upstream)
                  [0.2, 0.3]]

Backward — dL/dA = dL/dD @ B^T:
  B^T = [[5, 7],
         [6, 8]]

  dA = [[1.0, 0.5],   @ [[5, 7],   = [[1.0×5+0.5×6,  1.0×7+0.5×8],
        [0.2, 0.3]]      [6, 8]]      [0.2×5+0.3×6,  0.2×7+0.3×8]]

  dA = [[8.0,  11.0],
        [2.8,   3.8]]

Verify dL/da11 by hand:
  a11 appears in: d11 (coeff b11=5) and d12 (coeff b12=6)
  dL/da11 = dL/dd11 × 5 + dL/dd12 × 6 = 1.0×5 + 0.5×6 = 8.0 ✓

Backward — dL/dB = A^T @ dL/dD:
  A^T = [[1, 3],
         [2, 4]]

  dB = [[1, 3],  @ [[1.0, 0.5],  = [[1.6, 1.4],
        [2, 4]]     [0.2, 0.3]]     [2.8, 2.2]]

Backward — dL/dC = dL/dD.sum(axis=0):
  dC = [1.0+0.2, 0.5+0.3] = [1.2, 0.8]

  (C was broadcast to both rows, so gradients from both rows add up)
```

### 3.11 dhpreact — Backprop Through tanh

`h = tanh(hpreact)`. The local derivative of tanh is `1 - tanh(x)^2 = 1 - h^2`:

```python
dhpreact = (1.0 - h ** 2) * dh                  # (32, 64)
```

```
tanh backward:

hpreact: [ 0.5,  -1.2,   0.8,  ...]     (inputs)
h:       [ 0.46, -0.83,  0.66, ...]     (outputs = tanh(hpreact))
1 - h²:  [ 0.79,  0.31,  0.56, ...]     (local gradient)
dh:      [ 0.02, -0.01,  0.03, ...]     (incoming gradient)
dhpreact:[ 0.016,-0.003, 0.017, ...]     (local × incoming)
           ↑                ↑
       good gradient    still decent
       (h ≈ 0.46)      (h ≈ 0.66)
```

### 3.12 dbngain, dbnraw, dbnbias — Backprop Through BatchNorm Scale/Shift

`hpreact = bngain * bnraw + bnbias` where bngain and bnbias are (1, 64), bnraw and hpreact are (32, 64).

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # (1, 64) — sum undoes broadcast
dbnraw = bngain * dhpreact                            # (32, 64)
dbnbias = dhpreact.sum(0, keepdim=True)               # (1, 64) — sum undoes broadcast
```

The pattern: `bngain` (1, 64) was broadcast vertically to (32, 64) in the forward pass, so in the backward pass we sum across dimension 0 to collapse back to (1, 64).

**Concrete example** (3 examples, 2 neurons):

```
Forward:
bngain (1 × 2): [γ1, γ2] = [1.5, 0.8]     ← learnable scale
bnbias (1 × 2): [β1, β2] = [0.1, -0.2]    ← learnable shift
bnraw  (3 × 2): [[ 0.3, -1.0],             ← normalized inputs
                  [-0.5,  0.7],
                  [ 1.2,  0.2]]

hpreact = bngain * bnraw + bnbias:

  bngain broadcasts:   [[1.5, 0.8],         bnbias broadcasts: [[0.1, -0.2],
                         [1.5, 0.8],    ×    bnraw     +         [0.1, -0.2],
                         [1.5, 0.8]]                              [0.1, -0.2]]

  hpreact = [[1.5×0.3+0.1,  0.8×(-1.0)+(-0.2)],   = [[ 0.55, -1.0 ],
             [1.5×(-0.5)+0.1, 0.8×0.7+(-0.2)  ],     [-0.65,  0.36],
             [1.5×1.2+0.1,  0.8×0.2+(-0.2)    ]]     [ 1.90, -0.04]]

Backward (suppose dhpreact):
dhpreact = [[ 0.1,  0.2],
            [-0.3,  0.1],
            [ 0.4, -0.1]]

dbngain = sum over examples of (bnraw * dhpreact):
  neuron 0: 0.3×0.1 + (-0.5)×(-0.3) + 1.2×0.4 = 0.03 + 0.15 + 0.48 = 0.66
  neuron 1: (-1.0)×0.2 + 0.7×0.1 + 0.2×(-0.1) = -0.20 + 0.07 - 0.02 = -0.15
  → dbngain = [[0.66, -0.15]]    shape (1, 2) ✓

dbnraw = bngain * dhpreact:      (bngain broadcasts down)
  = [[1.5×0.1,  0.8×0.2 ],    = [[ 0.15, 0.16],
     [1.5×(-0.3), 0.8×0.1],      [-0.45, 0.08],
     [1.5×0.4, 0.8×(-0.1)]]      [ 0.60,-0.08]]   shape (3, 2) ✓

dbnbias = sum over examples of dhpreact:
  neuron 0: 0.1 + (-0.3) + 0.4 = 0.2
  neuron 1: 0.2 + 0.1 + (-0.1) = 0.2
  → dbnbias = [[0.2, 0.2]]      shape (1, 2) ✓

  (bias gradient = just sum the incoming gradients, same as in linear layers)
```

### 3.13 dbndiff, dbnvar_inv — Backprop Through Normalization Multiply

`bnraw = bndiff * bnvar_inv` where bndiff is (32, 64) and bnvar_inv is (1, 64).

```python
dbndiff = bnvar_inv * dbnraw                          # (32, 64)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)   # (1, 64)
```

Note: `dbndiff` is incomplete — `bndiff` is also used to compute `bndiff2`, so a second branch contributes later.

### 3.14 dbnvar — Backprop Through Power of -0.5

`bnvar_inv = (bnvar + 1e-5) ** (-0.5)`. Using the power rule: d/dx of x^(-0.5) = -0.5 · x^(-1.5):

```python
dbnvar = (-0.5 * (bnvar + 1e-5) ** (-1.5)) * dbnvar_inv   # (1, 64)
```

### 3.15 Bessel's Correction Aside

The code uses `1/(n-1)` instead of `1/n` for the variance:

```python
bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)    # unbiased estimate
```

**Why**: Mini-batches are small samples from a larger population. Dividing by `n` systematically underestimates the true variance (biased estimator). Dividing by `n-1` corrects this (Bessel's correction).

**The BatchNorm paper's inconsistency**: The original paper uses `1/n` during training but `1/(n-1)` for the running variance used at inference — a train/test mismatch. PyTorch's `BatchNorm1d` follows this inconsistency. Karpathy considers this a bug and prefers `1/(n-1)` consistently.

### 3.16 dbndiff2 — Backprop Through Sum (with broadcasting)

`bnvar = bndiff2.sum(0, keepdim=True) / (n-1)` where bndiff2 is (32, 64) and bnvar is (1, 64).

Forward: sum collapses 32 rows into 1 row. Backward: gradient replicates from 1 row to 32 rows.

```python
dbndiff2 = torch.ones_like(bndiff2) * (1.0 / (n - 1)) * dbnvar   # (32, 64)
# dbnvar (1,64) broadcasts to (32,64)
```

### 3.17 dbndiff (second branch) — Backprop Through Square

`bndiff2 = bndiff ** 2`. Derivative of x^2 is 2x:

```python
dbndiff += 2 * bndiff * dbndiff2                 # (32, 64) — += for second branch!
```

### 3.18 dhprebn, dbnmeani — Backprop Through Subtraction with Broadcasting

`bndiff = hprebn - bnmeani` where hprebn is (32, 64) and bnmeani is (1, 64).

```python
dhprebn = dbndiff.clone()                         # (32, 64) — gradient passes through
dbnmeani = (-dbndiff).sum(0, keepdim=True)        # (1, 64) — negative, then sum
```

Note: `dhprebn` is incomplete — `bnmeani` depends on `hprebn`, creating a second branch.

### 3.19 dhprebn (second branch) — Backprop Through Mean

`bnmeani = hprebn.sum(0, keepdim=True) / n`. Backward: scale by `1/n` and replicate:

```python
dhprebn += torch.ones_like(hprebn) * (1.0 / n) * dbnmeani   # (32, 64)
```

### 3.20 dembcat, dW1, db1 — Backprop Through Linear Layer 1

`hprebn = embcat @ W1 + b1` — same pattern as Layer 2:

```python
dembcat = dhprebn @ W1.T                         # (32, 30)
dW1 = embcat.T @ dhprebn                         # (30, 64)
db1 = dhprebn.sum(0)                             # (64,)
```

### 3.21 demb — Backprop Through View

`embcat = emb.view(emb.shape[0], -1)` just reshapes (32, 3, 10) → (32, 30). No math, just reshape back:

```python
demb = dembcat.view(emb.shape)                   # (32, 3, 10)
```

### 3.22 dC — Backprop Through Embedding Lookup

`emb = C[Xb]` plucks rows from C (27, 10) into emb (32, 3, 10). The backward pass deposits gradients back to the correct rows:

```python
dC = torch.zeros_like(C)                         # (27, 10)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]       # += because the same row may be used many times
```

```
Forward: pluck rows from C             Backward: deposit gradients back

C (27 × 10):                           dC (27 × 10):
row 0: [...]  ← used by Xb[0,0]=0     row 0: += demb[0,0] + demb[2,1] + ...
row 1: [...]  ← used by Xb[0,1]=1                (sum of all uses)
row 2: [...]                           row 2: += 0  (not used → stays zero)
 ...                                    ...
row 4: [...]  ← used by Xb[0,2]=4     row 4: += demb[0,2] + ...

Key: if a row is used N times in the batch,
     N gradient vectors get summed into that row of dC
```

**Concrete example** (3 examples, vocab of 4, embedding dim 2, block_size 2):

```
C (4 × 2):                 Xb (3 × 2):
  char 0 (.): [0.5, 0.1]     ex 0: [0, 2]    ← chars '.', 'b'
  char 1 (a): [0.3, 0.8]     ex 1: [2, 1]    ← chars 'b', 'a'
  char 2 (b): [0.7, 0.4]     ex 2: [0, 0]    ← chars '.', '.'
  char 3 (c): [0.2, 0.9]

Forward: emb = C[Xb]   →   emb (3 × 2 × 2):
  emb[0,0] = C[0] = [0.5, 0.1]     emb[0,1] = C[2] = [0.7, 0.4]
  emb[1,0] = C[2] = [0.7, 0.4]     emb[1,1] = C[1] = [0.3, 0.8]
  emb[2,0] = C[0] = [0.5, 0.1]     emb[2,1] = C[0] = [0.5, 0.1]

Suppose demb (3 × 2 × 2):
  demb[0,0] = [0.1, 0.2]    demb[0,1] = [0.3, 0.1]
  demb[1,0] = [0.4, 0.5]    demb[1,1] = [0.2, 0.3]
  demb[2,0] = [0.1, 0.1]    demb[2,1] = [0.6, 0.2]

Backward: deposit demb back into dC, summing duplicates:

  dC[0] ← used at [0,0], [2,0], [2,1]:
    = demb[0,0] + demb[2,0] + demb[2,1]
    = [0.1,0.2] + [0.1,0.1] + [0.6,0.2] = [0.8, 0.5]

  dC[1] ← used at [1,1]:
    = demb[1,1] = [0.2, 0.3]

  dC[2] ← used at [0,1], [1,0]:
    = demb[0,1] + demb[1,0]
    = [0.3,0.1] + [0.4,0.5] = [0.7, 0.6]

  dC[3] ← never used:
    = [0.0, 0.0]

Notice: char '.' (index 0) was used 3 times → its gradient is the
sum of 3 vectors. Char 'c' (index 3) was unused → zero gradient.
```

### 3.23 Complete Backprop Map

Every operation in the forward pass, its backward rule, and how the gradient flows through each step. Read the backward column from bottom to top — each step receives the gradient computed by the step below it.

```
FORWARD PASS (read top→bottom)            BACKWARD PASS (read bottom→top)
============================              ============================

─── Embedding ───────────────────────     ─── Embedding ─────────────────────────

emb = C[Xb]                              dC: scatter-add demb into rows
  (27,10) → (32,3,10)                      ▲ receives demb (32,3,10)
  Rule: index lookup                        Rule: deposit with +=
                                            Duality: pluck ↔ scatter
                        │                                     ▲
                        ▼                                     │
embcat = emb.view(32, -1)                demb = dembcat.view(32,3,10)
  (32,3,10) → (32,30)                      ▲ receives dembcat (32,30)
  Rule: reshape                             Rule: reverse reshape
                                            Duality: flatten ↔ unflatten
                        │                                     ▲
                        ▼                                     │
─── Layer 1 ─────────────────────────     ─── Layer 1 ───────────────────────────

hprebn = embcat @ W1 + b1                dembcat = dhprebn @ W1.T        (32,30)
  (32,30)@(30,64)+(64,) → (32,64)        dW1 = embcat.T @ dhprebn       (30,64)
  Rule: matmul + bias                      db1 = dhprebn.sum(0)          (64,)
                                            ▲ receives dhprebn (32,64)
                                            Duality: A@B ↔ dOut@B.T / A.T@dOut
                        │                                     ▲
                        ▼                                     │
─── BatchNorm ───────────────────────     ─── BatchNorm ─────────────────────────

bnmeani = hprebn.sum(0) / n              dhprebn  = dbndiff.clone()
  (32,64) → (1,64)                        dhprebn += (1/n) * dbnmeani
  Rule: mean (sum + scale)                  ▲ receives dbndiff (32,64) + dbnmeani (1,64)
                                            Duality: sum ↔ replicate
                        │                                     ▲
                        ▼                                     │
bndiff = hprebn - bnmeani                dbndiff (has 2 branches, both += into it)
  (32,64) - (1,64) → (32,64)              dbnmeani = (-dbndiff).sum(0, keepdim=True)
  Rule: subtract with broadcast             ▲ receives dbndiff (32,64)
                                            Duality: broadcast ↔ sum
                        │                                     ▲
                        ▼                               ┌─────┴─────┐
bndiff2 = bndiff ** 2                    │ branch 2:                │ branch 1:
  (32,64) → (32,64)                      │ dbndiff += 2*bndiff      │ dbndiff = bnvar_inv
  Rule: element-wise square               │            *dbndiff2     │            *dbnraw
                        │                 │ ▲ receives dbndiff2      │ ▲ receives dbnraw
                        ▼                 │              ▲           │              ▲
bnvar = bndiff2.sum(0)/(n-1)            │ dbndiff2 = (1/(n-1))     │              │
  (32,64) → (1,64)                      │            *dbnvar        │              │
  Rule: sum + scale                      │ ▲ receives dbnvar        │              │
                        │                 │              ▲           │              │
                        ▼                 │              │           │              │
bnvar_inv = (bnvar+ε)^(-0.5)           │ dbnvar = -0.5*(bnvar+ε)  │              │
  (1,64) → (1,64)                      │          ^(-1.5)          │              │
  Rule: power                           │          *dbnvar_inv      │              │
                        │                 │ ▲ receives dbnvar_inv    │              │
                        ▼                 │              ▲           │              │
bnraw = bndiff * bnvar_inv              │ dbnvar_inv = (bndiff      │ dbnraw = bnvar_inv
  (32,64) * (1,64) → (32,64)           │   *dbnraw).sum(0)         │            *dbnraw
  Rule: multiply + broadcast             └──────────────┬───────────┘
                                            ▲ receives dbnraw (32,64)
                        │                                     ▲
                        ▼                                     │
hpreact = bngain*bnraw + bnbias          dbngain = (bnraw*dhpreact).sum(0)  (1,64)
  (1,64)*(32,64)+(1,64) → (32,64)        dbnbias = dhpreact.sum(0)         (1,64)
  Rule: scale + shift + broadcast          dbnraw = bngain * dhpreact       (32,64)
                                            ▲ receives dhpreact (32,64)
                                            Duality: broadcast ↔ sum
                        │                                     ▲
                        ▼                                     │
─── Activation ──────────────────────     ─── Activation ────────────────────────

h = tanh(hpreact)                        dhpreact = (1 - h²) * dh
  (32,64) → (32,64)                        ▲ receives dh (32,64)
  Rule: element-wise activation             Rule: local deriv × incoming
                        │                                     ▲
                        ▼                                     │
─── Layer 2 ─────────────────────────     ─── Layer 2 ───────────────────────────

logits = h @ W2 + b2                     dh = dlogits @ W2.T               (32,64)
  (32,64)@(64,27)+(27,) → (32,27)        dW2 = h.T @ dlogits              (64,27)
  Rule: matmul + bias                      db2 = dlogits.sum(0)            (27,)
                                            ▲ receives dlogits (32,27)
                                            Duality: A@B ↔ dOut@B.T / A.T@dOut
                        │                                     ▲
                        ▼                                     │
─── Softmax + Cross-Entropy Loss ────     ─── Softmax + Cross-Entropy Loss ─────

logit_maxes = logits.max(1)              dlogits += one_hot(max_idx)*dlogit_maxes (≈0)
  (32,27) → (32,1)                         ▲ receives dlogit_maxes (32,1)
  Rule: pluck max per row
                        │                                     ▲
                        ▼                                     │
norm_logits = logits - logit_maxes       dlogits = dnorm_logits.clone()
  (32,27) - (32,1) → (32,27)              dlogit_maxes = (-dnorm_logits).sum(1)
  Rule: subtract + broadcast                ▲ receives dnorm_logits (32,27)
                                            Duality: broadcast ↔ sum
                        │                                     ▲
                        ▼                                     │
counts = exp(norm_logits)                dnorm_logits = counts * dcounts
  (32,27) → (32,27)                        ▲ receives dcounts (32,27)
  Rule: element-wise exp                    Rule: d/dx e^x = e^x
                        │                                     ▲
                        ▼                               ┌─────┴─────┐
counts_sum = counts.sum(1)               │ branch 2:                │ branch 1:
  (32,27) → (32,1)                       │ dcounts += dcounts_sum   │ dcounts = counts_sum_inv
  Rule: sum per row                       │  (broadcast)             │            * dprobs
                        │                 │ ▲ receives dcounts_sum   │ ▲ receives dprobs
                        ▼                 │              ▲           │              ▲
counts_sum_inv = counts_sum^(-1)         │ dcounts_sum = -counts_sum │              │
  (32,1) → (32,1)                       │  ^(-2)*dcounts_sum_inv    │              │
  Rule: inverse                          │ ▲ receives dcounts_sum_inv              │
                        │                 │              ▲           │              │
                        ▼                 │              │           │              │
probs = counts * counts_sum_inv          │ dcounts_sum_inv = (counts │              │
  (32,27) * (32,1) → (32,27)           │  *dprobs).sum(1)          │              │
  Rule: multiply + broadcast             └──────────────┬───────────┘
                                            ▲ receives dprobs (32,27)
                        │                                     ▲
                        ▼                                     │
logprobs = log(probs)                    dprobs = (1/probs) * dlogprobs
  (32,27) → (32,27)                        ▲ receives dlogprobs (32,27)
  Rule: element-wise log                    Rule: d/dx log(x) = 1/x
                        │                                     ▲
                        ▼                                     │
loss = -logprobs[range(n),Yb].mean()     dlogprobs = zeros; dlogprobs[range(n),Yb] = -1/n
  (32,27) → scalar                         ▲ STARTS HERE — loss is the root
  Rule: index + negate + mean               dloss/dloss = 1.0 (trivially)

The gradient originates at the loss (bottom-right) and flows upward
through every operation, accumulating via the chain rule at each step.
Each "▲ receives ..." shows what gradient arrives from the step below.
```

---

## 4. Exercise 2: Efficient Cross-Entropy Backward

### 4.1 The Problem with Atomic Backprop

In Exercise 1, we backpropagated through ~10 operations to get from `loss` to `dlogits`. This is correct but wasteful. The mathematical expression for cross-entropy has a much simpler derivative.

### 4.2 Deriving the Analytical Gradient

Starting from the loss for a single example:

```
loss_i = -log( P(y_i) )

where P(y_i) = exp(L_{y_i}) / Σ_j exp(L_j)    (softmax)
```

**Step 1**: Simplify the expression. Plug softmax into the log:

```
loss_i = -log( exp(L_y) / Σ_j exp(L_j) )

       = -[ log(exp(L_y)) - log(Σ_j exp(L_j)) ]     ← log(a/b) = log(a) - log(b)

       = -L_y + log(Σ_j exp(L_j))                    ← log(exp(x)) = x
```

This is much cleaner: the loss is just the negative logit of the correct class plus the log-sum-exp of all logits.

**Step 2**: Differentiate with respect to logit `L_k`.

The second term `log(Σ_j exp(L_j))` is the same for both cases. Its derivative:

```
∂/∂L_k [ log(Σ_j exp(L_j)) ] = exp(L_k) / Σ_j exp(L_j) = P(k)    ← this is just softmax!
```

(This uses the chain rule: derivative of log(u) is 1/u, times derivative of u with respect to L_k which is exp(L_k).)

**Case 1: k = y_i** (the correct class)

```
∂loss_i / ∂L_k = ∂/∂L_k [ -L_y ] + P(k)
               =         -1       + P(k)
               = P(k) - 1
```

**Case 2: k ≠ y_i** (any incorrect class)

```
∂loss_i / ∂L_k = ∂/∂L_k [ -L_y ] + P(k)
               =          0       + P(k)       ← -L_y doesn't depend on L_k when k ≠ y
               = P(k)
```

Combined: the gradient is just the softmax probabilities, minus 1 at the correct class position.

**Concrete numerical example** (4 classes, correct label y = 2):

```
logits:   [1.0,  2.0,  0.5,  1.5]
exp():    [2.72, 7.39, 1.65, 4.48]    sum = 16.24
softmax:  [0.17, 0.46, 0.10, 0.28]    sum = 1.00

Gradient for each logit:
  L_0: k≠y → dL/dL_0 = P(0) = 0.17          (push down)
  L_1: k≠y → dL/dL_1 = P(1) = 0.46          (push down strongly — high prob wrong class)
  L_2: k=y → dL/dL_2 = P(2) - 1 = -0.90     (push UP — correct class needs boosting)
  L_3: k≠y → dL/dL_3 = P(3) = 0.28          (push down)

Check: 0.17 + 0.46 + (-0.90) + 0.28 = 0.01 ≈ 0  ✓ (forces balance)
```

For a batch of n examples averaged: divide everything by n.

### 4.3 Implementation

```python
dlogits = F.softmax(logits, dim=1)               # (32, 27) — softmax probabilities
dlogits[range(n), Yb] -= 1                        # subtract 1 at correct positions
dlogits /= n                                      # average over batch
```

This replaces ~10 lines of backward code with 3 lines, and is both faster and numerically more stable.

### 4.4 Visual Intuition: Push and Pull Forces

```
dlogits for one example (n=1 for clarity):

logits:  [ 0.2,  1.5,  -0.3,  0.1, ...]    y = 1 (correct class)
softmax: [0.05, 0.18,  0.03, 0.04, ...]    sum = 1.0
dlogits: [0.05, 0.18-1, 0.03, 0.04, ...]
       = [0.05, -0.82,  0.03, 0.04, ...]

         ↑pull  ↑PUSH   ↑pull  ↑pull
         down    UP      down   down

Sum of dlogits = 0  (forces balance exactly)
```

**Interpretation**:

- Every incorrect class gets a **positive gradient** (proportional to its probability) → push its logit **down**
- The correct class gets a **negative gradient** (probability - 1) → push its logit **up**
- The forces are proportional to the model's current confidence: the more confidently wrong the prediction, the stronger the corrective force
- If the prediction is already perfect (P(correct) = 1.0), all gradients are zero — no push or pull needed

Think of the neural network as a massive pulley system. The dlogits are the tension at the top, pulling up on the correct answer and pulling down on incorrect answers. This tension propagates backward through the entire network, gently tugging every weight and bias in the right direction.

**How gradient strength depends on prediction quality** (3 classes, correct = class 0):

```
Scenario A: model is badly wrong (P(correct) = 0.05)
  softmax:  [0.05,  0.70,  0.25]
  dlogits:  [-0.95, 0.70,  0.25]
             ↑ strong pull UP    ↑ strong push down

Scenario B: model is uncertain (P(correct) = 0.40)
  softmax:  [0.40,  0.35,  0.25]
  dlogits:  [-0.60, 0.35,  0.25]
             ↑ moderate pull     ↑ moderate push

Scenario C: model is nearly perfect (P(correct) = 0.95)
  softmax:  [0.95,  0.03,  0.02]
  dlogits:  [-0.05, 0.03,  0.02]
             ↑ tiny pull         ↑ tiny push

Scenario D: model is perfect (P(correct) = 1.00)
  softmax:  [1.00,  0.00,  0.00]
  dlogits:  [ 0.00, 0.00,  0.00]    ← all zeros! nothing to learn.

The worse the prediction, the larger the gradient forces.
The better the prediction, the more the gradients vanish.
This is self-regulating: the network automatically focuses
its learning effort on the examples it gets wrong.
```

---

## 5. Exercise 3: Efficient Batch Normalization Backward

### 5.1 The BatchNorm Forward Pass (Compact)

```python
# Forward (single formula)
bnmeani = hprebn.sum(0, keepdim=True) / n
bndiff = hprebn - bnmeani
bnvar = (bndiff ** 2).sum(0, keepdim=True) / (n - 1)
bnvar_inv = (bnvar + 1e-5) ** (-0.5)
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
```

We want to go directly from `dhpreact` to `dhprebn` in a single analytical formula, ignoring gamma/beta (they're trivial).

### 5.2 Setting Up the Math

Using the BatchNorm paper's notation:

```
μ = (1/m) Σᵢ xᵢ                            mean
σ² = (1/(m-1)) Σᵢ (xᵢ - μ)²                variance (with Bessel's)
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)                 normalized
yᵢ = γ · x̂ᵢ + β                             output
```

Given `dL/dyᵢ` for all i, we want `dL/dxᵢ` for all i.

### 5.3 The Computational Graph

```
x₁, x₂, ..., xₘ    (m inputs, e.g., 32 examples for one neuron)
    |     |     |
    v     v     v
    └─────┼─────┘
          v
    μ (scalar)  ←── mean of all xᵢ
    |     |
    v     v
   x₁-μ, x₂-μ, ..., xₘ-μ    (bndiff)
    |  \    |  \        |  \
    v   v   v   v       v   v
   x̂₁  (x₁-μ)²  x̂₂  (x₂-μ)²  ...
    |      |       |      |
    v      └───────┼──────┘
    |              v
    |         σ² (scalar)  ←── variance
    |              |
    |              v
    |         (σ²+ε)^(-0.5)  ←── bnvar_inv
    |              |
    └──────── × ──┘
              |
          x̂₁, x̂₂, ..., x̂ₘ   (bnraw)
              |
          γ · x̂ᵢ + β = yᵢ     (hpreact)
```

**Key complexity**: each `xᵢ` fans out to 3 destinations:

1. Its own `x̂ᵢ` (parallel, independent arrows) — "what is my own normalized value?"
2. μ (all xᵢ contribute to the shared mean) — "what should we subtract from everyone?"
3. σ² (all xᵢ contribute to the shared variance) — "what should we divide everyone by?"

**Intuition**: this is what makes BatchNorm tricky. In a simple element-wise operation, each input only affects its own output. But in BatchNorm, changing one example's input changes the mean and variance, which changes **every** example's output. Every example is mathematically entangled with every other example in the batch.

### 5.4 Step-by-Step Derivation

**Step 1**: `dL/dx̂ᵢ = dL/dyᵢ · γ` (trivial — gamma is just a scale factor)

**Step 2**: `dL/dσ²`

σ² is a single scalar used by all x̂ᵢ, so sum over all i:

```
dL/dσ² = Σᵢ (dL/dx̂ᵢ) · (∂x̂ᵢ/∂σ²)

∂x̂ᵢ/∂σ² = (xᵢ - μ) · (-1/2) · (σ² + ε)^(-3/2)

dL/dσ² = -1/2 · (σ² + ε)^(-3/2) · Σᵢ (dL/dx̂ᵢ) · (xᵢ - μ)
```

**Step 3**: `dL/dμ`

μ flows to all x̂ᵢ (m arrows) and also to σ² (1 arrow), totaling m+1 arrows.

```
dL/dμ = Σᵢ (dL/dx̂ᵢ) · (∂x̂ᵢ/∂μ) + (dL/dσ²) · (∂σ²/∂μ)
```

**Critical simplification**: When μ is the actual mean of the xᵢ, the term `∂σ²/∂μ = (2/(m-1)) Σᵢ (xᵢ - μ)` equals zero because the sum of deviations from the mean is always zero. The entire second term vanishes!

```
Why Σ(xᵢ - μ) = 0 always:

Example: x = [2, 5, 8, 1]    μ = (2+5+8+1)/4 = 4.0

  x₁ - μ = 2 - 4 = -2
  x₂ - μ = 5 - 4 =  1
  x₃ - μ = 8 - 4 =  4
  x₄ - μ = 1 - 4 = -3
                    ----
  sum            =   0   ← always! by definition of the mean.

This is not a coincidence — it's a mathematical identity.
The mean is the "balance point" where positive and negative
deviations exactly cancel. So ∂σ²/∂μ = 0, and the entire
second term in dL/dμ disappears. This is the key simplification
that makes the BatchNorm backward formula manageable.
```

```
dL/dμ = -(σ² + ε)^(-1/2) · Σᵢ (dL/dx̂ᵢ)
```

**Step 4**: `dL/dxᵢ`

Each xᵢ has 3 outgoing arrows: to x̂ᵢ, to μ, and to σ². Sum all three:

```
dL/dxᵢ = (dL/dx̂ᵢ) · (∂x̂ᵢ/∂xᵢ) + (dL/dμ) · (∂μ/∂xᵢ) + (dL/dσ²) · (∂σ²/∂xᵢ)
```

Where:

- `∂x̂ᵢ/∂xᵢ = (σ² + ε)^(-1/2)` = bnvar_inv
- `∂μ/∂xᵢ = 1/m`
- `∂σ²/∂xᵢ = 2(xᵢ - μ) / (m-1)`

### 5.5 Final Simplified Formula

After plugging everything in, factoring out `(σ² + ε)^(-1/2)`, and simplifying:

```
dL/dxᵢ = (σ² + ε)^(-1/2) / (m-1) · [ (m-1) · dL/dx̂ᵢ  -  Σⱼ dL/dx̂ⱼ  -  x̂ᵢ · Σⱼ (dL/dx̂ⱼ · x̂ⱼ) ]
                                        ─────────────     ──────────     ─────────────────────────
                                          Term 1            Term 2               Term 3
```

**Intuition for each term**:

- **Term 1**: `(m-1) · dL/dx̂ᵢ` — the **direct gradient**. This is what you'd get if μ and σ² were just constants that don't depend on x. It's the "naive" gradient that ignores the entanglement.

- **Term 2**: `- Σⱼ dL/dx̂ⱼ` — the **mean correction**. Changing xᵢ shifts the mean μ, which shifts *every* example's normalized output. This term subtracts out the average gradient across the batch, ensuring the mean of the gradients is zero. (If the gradient wanted all values to go up, this term says "you can't all go up — the mean is fixed.")

- **Term 3**: `- x̂ᵢ · Σⱼ (dL/dx̂ⱼ · x̂ⱼ)` — the **variance correction**. Changing xᵢ changes the variance σ², which rescales *every* example. This correction is proportional to x̂ᵢ itself — values far from the mean (large |x̂ᵢ|) get a larger correction because they contribute more to the variance.

```
Intuition summary:
  "raw gradient"  −  "mean shift effect"  −  "variance scale effect"

If BatchNorm didn't couple examples (if μ and σ² were fixed constants),
only Term 1 would exist. Terms 2 and 3 are the "correction taxes" you
pay because changing one example's input changes every example's output.
```

### 5.6 Implementation

```python
# dhpreact is the incoming gradient, shape (32, 64)
# bnraw is x̂, shape (32, 64)
# bnvar_inv is (σ² + ε)^(-0.5), shape (1, 64)

dhprebn = (
    bnvar_inv / (n - 1) * (
        (n - 1) * bngain * dhpreact
        - (bngain * dhpreact).sum(0)
        - bnraw * (bngain * dhpreact * bnraw).sum(0)
    )
)
```

```
Batch normalization backward — single formula:

dhpreact (32, 64)    ← incoming gradient
        |
        | three terms, all operating per-neuron (column):
        |
        | Term 1: (m-1) · γ · dhpreact         ← scale the per-example gradient
        | Term 2: - Σⱼ(γ · dhpreact_j)         ← subtract the column sum (mean correction)
        | Term 3: - x̂ · Σⱼ(γ · dhpreact_j · x̂ⱼ) ← subtract variance correction
        |
        | All three combined, scaled by bnvar_inv / (m-1)
        v
dhprebn (32, 64)     ← output gradient
```

This replaces the ~15 lines of atomic backprop with a single expression, and it processes all 64 neurons in parallel via broadcasting.

**Concrete numerical walkthrough** (4 examples, 1 neuron, γ=1 for simplicity):

```
Forward:
  x = [2.0, 6.0, 4.0, 8.0]        ← 4 examples, 1 neuron
  μ = (2+6+4+8)/4 = 5.0
  σ² = [(2-5)² + (6-5)² + (4-5)² + (8-5)²] / (4-1)
     = [9 + 1 + 1 + 9] / 3 = 6.667
  bnvar_inv = (6.667 + 0.00001)^(-0.5) = 0.387
  x̂ = (x - μ) × bnvar_inv
     = [-3, 1, -1, 3] × 0.387 = [-1.161, 0.387, -0.387, 1.161]

Backward (suppose dL/dx̂ = [0.5, -0.2, 0.3, -0.1]):
  m = 4, ε ≈ 0

  Σⱼ dL/dx̂ⱼ = 0.5 + (-0.2) + 0.3 + (-0.1) = 0.5

  Σⱼ (dL/dx̂ⱼ · x̂ⱼ) = 0.5×(-1.161) + (-0.2)×0.387 + 0.3×(-0.387) + (-0.1)×1.161
                       = -0.581 + (-0.077) + (-0.116) + (-0.116) = -0.890

  For x₁ (i=1, x̂₁ = -1.161):
    Term 1: (4-1) × 0.5 = 1.5
    Term 2: -0.5
    Term 3: -(-1.161) × (-0.890) = -1.033

    dL/dx₁ = 0.387 / 3 × (1.5 - 0.5 - 1.033) = 0.129 × (-0.033) = -0.004

  For x₂ (i=2, x̂₂ = 0.387):
    Term 1: (4-1) × (-0.2) = -0.6
    Term 2: -0.5
    Term 3: -(0.387) × (-0.890) = 0.344

    dL/dx₂ = 0.387 / 3 × (-0.6 - 0.5 + 0.344) = 0.129 × (-0.756) = -0.098

Notice how the three terms interact:
  - Term 1 wants to pass the raw gradient through
  - Term 2 shifts everyone by the same amount (mean correction)
  - Term 3 gives different corrections to different examples
    based on how far they are from the mean (variance correction)
```

---

## 6. Exercise 4: Putting It All Together

### 6.1 The Complete Manual Backward Pass

With the efficient cross-entropy and BatchNorm backward passes, the entire backward pass for the two-layer MLP fits in ~20 lines:

```python
# --- Backward Pass (manual, no loss.backward()) ---

# Cross-entropy backward (Exercise 2)
dlogits = F.softmax(logits, dim=1)                #  ┐
dlogits[range(n), Yb] -= 1                        #  ├─ Loss → logits
dlogits /= n                                      #  ┘

# Layer 2 backward
dh = dlogits @ W2.T                              # (32, 64)  ┐
dW2 = h.T @ dlogits                              # (64, 27)  ├─ Linear layer 2
db2 = dlogits.sum(0)                             # (27,)     ┘

# tanh backward
dhpreact = (1.0 - h ** 2) * dh                   # (32, 64)  ── Activation

# BatchNorm backward (Exercise 3)
dbngain = (bnraw * dhpreact).sum(0, keepdim=True) #           ┐
dbnbias = dhpreact.sum(0, keepdim=True)           #           │
dhprebn = (bnvar_inv / (n - 1)) * (              #           ├─ BatchNorm
    (n - 1) * bngain * dhpreact                   #           │
    - (bngain * dhpreact).sum(0)                  #           │
    - bnraw * (bngain * dhpreact * bnraw).sum(0)  #           │
)                                                 #           ┘

# Layer 1 backward
dembcat = dhprebn @ W1.T                          # (32, 30)  ┐
dW1 = embcat.T @ dhprebn                          # (30, 64)  ├─ Linear layer 1
db1 = dhprebn.sum(0)                              # (64,)     ┘

# Embedding backward
demb = dembcat.view(emb.shape)                    # (32, 3, 10) ┐
dC = torch.zeros_like(C)                          #              │
for k in range(Xb.shape[0]):                      #              ├─ Embedding
    for j in range(Xb.shape[1]):                  #              │   (scatter)
        ix = Xb[k, j]                             #              │
        dC[ix] += demb[k, j]                      #              ┘
```

```
Visual map: gradient flow through the network (bottom to top = backward)

 loss (scalar)
   │
   ▼
 dlogits (32, 27)  ────────── Cross-entropy: softmax - one_hot
   │
   ├──▶ dW2 (64, 27)         Layer 2: h.T @ dlogits
   ├──▶ db2 (27,)             Layer 2: dlogits.sum(0)
   │
   ▼
 dh (32, 64)  ────────────── Layer 2: dlogits @ W2.T
   │
   ▼
 dhpreact (32, 64)  ──────── tanh: (1 - h²) × dh
   │
   ├──▶ dbngain (1, 64)      BatchNorm: (bnraw × dhpreact).sum(0)
   ├──▶ dbnbias (1, 64)      BatchNorm: dhpreact.sum(0)
   │
   ▼
 dhprebn (32, 64)  ────────── BatchNorm: single analytical formula
   │
   ├──▶ dW1 (30, 64)         Layer 1: embcat.T @ dhprebn
   ├──▶ db1 (64,)             Layer 1: dhprebn.sum(0)
   │
   ▼
 dembcat (32, 30)  ────────── Layer 1: dhprebn @ W1.T
   │
   ▼
 demb (32, 3, 10)  ────────── View: reshape back
   │
   ▼
 dC (27, 10)  ────────────── Embedding: scatter-add demb into rows
```

### 6.2 Training Without Autograd

```python
# Training loop — loss.backward() is GONE
with torch.no_grad():     # tell PyTorch we don't need autograd at all
    for i in range(max_steps):
        # ... mini-batch sampling ...
        # ... forward pass ...
        # ... manual backward pass (from above) ...

        # Parameter update
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad
```

**What `torch.no_grad()` does**: Normally, PyTorch builds a computational graph during every forward operation, recording what happened so it can later call `.backward()`. This costs memory and time. Since we're computing gradients manually, we tell PyTorch: "don't bother tracking anything — we'll handle it ourselves." This makes training faster and uses less memory.

**Verification**: Before removing `loss.backward()`, run both side by side for a few steps and confirm the gradients match:

```
Comparison after 1 training step:

Parameter     | loss.backward()  | manual backward  | max diff
------------- | ---------------- | ---------------- | --------
W2 grad[0,0]  | -0.00312         | -0.00312         | 0.0
b2 grad[0]    |  0.00847         |  0.00847         | 0.0
bngain grad   | -0.01193         | -0.01193         | 1e-9
W1 grad[0,0]  |  0.00056         |  0.00056         | 1e-9
C grad[0,0]   | -0.00204         | -0.00204         | 1e-9

The differences are either zero or on the order of 1e-9
(floating point rounding). The gradients are functionally identical.
```

The model achieves the same loss (~2.1) and produces the same quality samples as the autograd version. The only thing PyTorch provides now is `torch.tensor` for efficient computation — everything else is our own code.

**What we've achieved**: the full training loop — forward pass, loss computation, backward pass, parameter update — is now entirely transparent. There is no hidden magic. Every gradient that flows through this network is a line of code we wrote and understand.

---

## 7. Recurring Patterns and Rules

### 7.1 The Broadcasting Duality


| Forward pass                        | Backward pass                         | Example                           |
| ----------------------------------- | ------------------------------------- | --------------------------------- |
| **Sum** (collapse a dimension)      | **Replicate** (broadcast gradient)    | `counts_sum = counts.sum(1)`      |
| **Broadcast** (replicate along dim) | **Sum** (collapse gradient)           | `probs = counts * counts_sum_inv` |
| **Element-wise op**                 | Element-wise local derivative × chain | `counts = exp(norm_logits)`       |
| **Matrix multiply** `A @ B`         | `dA = dOut @ B.T`, `dB = A.T @ dOut`  | `logits = h @ W2`                 |
| **View/reshape**                    | Reverse reshape                       | `embcat = emb.view(N, -1)`        |
| **Indexing** (pluck elements)       | Scatter/deposit with `+=`             | `emb = C[Xb]`                     |


### 7.2 The Shape-Matching Trick for Matrix Multiply

You never need to memorize the matmul backward formulas. Just:

1. Know that `dA` must be the same shape as `A`
2. Know it must come from a matmul of `dOut` and `B` (possibly transposed)
3. There's only one arrangement of transposes that makes the shapes work
4. That arrangement is always correct

### 7.3 Multi-Use Variables

When a variable is used in multiple places (branches), its gradients from all branches must **add up**. In code, use `+=`:

```python
# dcounts has two branches:
dcounts = counts_sum_inv * dprobs              # branch 1
dcounts += torch.ones_like(counts) * dcounts_sum  # branch 2

# dhprebn has two branches:
dhprebn = dbndiff.clone()                      # branch 1 (from bndiff)
dhprebn += (1.0/n) * dbnmeani                  # branch 2 (from bnmeani)
```

This is the tensor-level version of what we learned in micrograd: when a node feeds into multiple consumers, gradients sum.

---

## 8. Key Takeaways

- `**loss.backward()` is not magic** — it performs exactly the chain rule operations we implemented manually, one tensor at a time
- **Matrix multiply backward is another matrix multiply** — just with transposes. Use shape-matching to derive which transpose.
- **Broadcasting forward = summation backward** (and vice versa). This duality appears everywhere.
- **Cross-entropy gradient** simplifies beautifully to `softmax(logits) - one_hot(labels)` scaled by `1/n`. Think of it as push/pull forces proportional to prediction error.
- **BatchNorm gradient** can be derived analytically into a single formula instead of backpropagating through ~8 atomic operations. The key simplification: `∂σ²/∂μ = 0` when μ is the actual mean.
- **Bessel's correction** (`1/(n-1)` vs `1/n`) matters for small batches. The original BatchNorm paper has a train/test mismatch here.
- **The entire backward pass** for a two-layer MLP with BatchNorm is ~20 lines of code
- **Numerical stability operations** (like subtracting logit_maxes) have near-zero gradients — they don't affect the loss, as expected

---

## 9. Looking Forward

The current MLP architecture is limited by its fixed context window (3 characters). The next step is **recurrent neural networks (RNNs)**, which can handle variable-length sequences. RNNs are essentially very deep networks when unrolled through time — making everything covered in Parts 3 and 4 (initialization, gradient flow, BatchNorm) even more critical.