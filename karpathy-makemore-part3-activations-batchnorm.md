# makemore Part 3: Activations, Gradients & Batch Normalization

**Source**: Andrej Karpathy's "makemore" series, Part 3
**Paper**: Ioffe & Szegedy (2015) — _Batch Normalization: Accelerating Deep Network Training_; He et al. (2015) — _Delving Deep into Rectifiers_
**Prerequisites**: MLP character-level language model from Part 2

---

## Overview

This lecture pauses before moving to recurrent neural networks to build a deep intuitive understanding of **activations and gradients** inside neural networks during training. The key insight: understanding how values flow forward and gradients flow backward is essential for training deep networks. The lecture covers three main areas: (1) fixing initialization problems that waste training time, (2) principled weight initialization via Kaiming init, and (3) Batch Normalization — the first major "modern innovation" that stabilized deep network training. Along the way, diagnostic tools for monitoring training health are introduced.

---

## 1. Fixing Initialization: The Softmax Confidence Problem

### 1.1 The Problem

At initialization, the first iteration records a loss of ~27. This is way too high. We can calculate what the loss **should** be:

- 27 possible characters, no reason to prefer any at init
- Expected probability for each character: `1/27`
- Expected loss: `-log(1/27) = 3.29`

```python
# What we expect at initialization
expected_loss = -torch.tensor(1.0 / 27).log()
# tensor(3.2958)
```

A loss of 27 means the network is **confidently wrong** — some logits are very large, creating extreme probabilities that happen to be assigned to incorrect characters.

### 1.2 Why It Happens

The logits are calculated as `h @ W2 + b2`. If `W2` and `b2` are initialized with standard random values, the logits take on extreme values, creating "fake confidence" in wrong answers.

```python
# 4-dimensional toy example
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = torch.softmax(logits, dim=0)
# tensor([0.25, 0.25, 0.25, 0.25]) — uniform, loss = 1.38 ✓

logits = torch.tensor([-2.0, 4.0, -1.0, 0.5])
probs = torch.softmax(logits, dim=0)
# One probability dominates — if it's wrong, loss is very high ✗
```

```
Case 1: logits ≈ 0 (good init)        Case 2: logits are extreme (bad init)

logits: [0.0,  0.0,  0.0,  0.0]       logits: [-2.0,  4.0, -1.0,  0.5]
           ↓ softmax                              ↓ softmax
probs:  [0.25, 0.25, 0.25, 0.25]      probs:  [0.002, 0.952, 0.006, 0.029]
                                                        ↑
If label = 2:                          If label = 2:  this dominates but
  loss = -log(0.25) = 1.38 ✓            loss = -log(0.006) = 5.1 ✗
  (expected, reasonable)                 (confidently wrong!)
```

### 1.3 The Fix

Make logits approximately zero at initialization:

```python
# Set b2 to zero (we don't want random bias on the output)
b2 = torch.randn(vocab_size) * 0       # effectively zeros

# Scale W2 very small (not exactly zero — keep a bit of entropy for symmetry breaking)
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
```

```
Before fix:                          After fix:

W2 (200, 27):                       W2 (200, 27):
  [-1.2,  0.8,  2.1, ...]             [-0.012, 0.008, 0.021, ...]
  [ 0.5, -0.3,  1.7, ...]             [ 0.005,-0.003, 0.017, ...]
  values ~ N(0, 1)                     values ~ N(0, 0.01)   ← tiny!

b2 (27,):                           b2 (27,):
  [0.4, -1.1, 0.7, ...]               [0, 0, 0, ...]         ← zeros

logits = h @ W2 + b2:               logits = h @ W2 + b2:
  [-5.2, 8.1, -3.4, ...]              [-0.05, 0.08, -0.03, ...]
  → extreme → loss ≈ 27               → near zero → loss ≈ 3.29 ✓
```

**Why not set W2 to exactly zero?** Setting weights to exactly zero breaks symmetry in problematic ways (covered later with dead neurons). Using a very small value like 0.01 keeps a tiny bit of randomness while making logits near-zero.

### 1.4 Impact

| Configuration            | Validation Loss |
| ------------------------ | --------------- |
| Original (random init)   | 2.17            |
| Fixed softmax confidence | **2.13**        |

The "hockey stick" shape in the loss curve disappears — the network no longer wastes the first few thousand iterations just learning to squash extreme logits down. Training time is spent on actual useful learning from the start.

---

## 2. Fixing Initialization: tanh Saturation

### 2.1 The Problem

Even with fixed logits, the hidden layer activations `h` have a deeper problem. Visualizing `h`:

```python
# Most values are ±1 — tanh is fully saturated
plt.hist(h.view(-1).tolist(), bins=50)
```

The histogram shows nearly all values pushed to -1 or +1. The pre-activations feeding into tanh are too large (ranging from -15 to +15), so tanh squashes everything to its flat tails.

### 2.2 Why This Kills Gradients

Recall from micrograd how tanh backward works:

```python
# In the backward pass of tanh:
# t = tanh(x)  — the output, between -1 and 1
# local gradient = 1 - t**2

# When t ≈ 1:   1 - 1**2 = 0   → gradient is ZERO
# When t ≈ -1:  1 - (-1)**2 = 0 → gradient is ZERO
# When t ≈ 0:   1 - 0**2 = 1   → gradient passes through fully
```

```
tanh function and its gradient:

output t
  1 |          ___________     ← flat tail: gradient ≈ 0
    |        /
    |      /                   ← steep middle: gradient ≈ 1
  0 |----/
    |  /
    |/
 -1 |___________               ← flat tail: gradient ≈ 0
    └──────────────── input x
     -5     0     5

local gradient (1 - t²):
  1 |        ∧
    |       / \
    |      /   \               ← only near t=0 does gradient survive
    |     /     \
  0 |____/       \____
    └──────────────── t
     -1     0     1
```

When the output of tanh is in the flat tails (±1), the gradient gets **multiplied by zero**. The gradient is destroyed — it can't flow backward through this layer. This is called the **vanishing gradient problem**.

### 2.3 Visualizing Saturation

```python
# Check what fraction of activations are in the "dead zone"
saturation = (h.abs() > 0.99).float().mean()
# If this is high (e.g., 90%), most neurons aren't learning
```

```
Saturation grid: |h| > 0.99?     (32 examples × 200 neurons)

         neuron 0  neuron 1  neuron 2  neuron 3 ... neuron 199
ex  0:     ■         □         □         ■            □
ex  1:     □         □         □         ■            □
ex  2:     ■         □         □         ■            ■
ex  3:     □         □         □         ■            □
 ...       ...       ...       ...       ...          ...
ex 31:     ■         □         □         ■            □

           ok        ok        ok      DEAD!          ok
         (mixed)   (mixed)   (mixed)  (all □)       (mixed)

■ = saturated (|h| > 0.99, gradient ≈ 0)
□ = active    (|h| < 0.99, gradient flows)

Neuron 3 is all ■ → dead neuron! No example ever activates it.
```

You can visualize this as a 2D boolean grid (32 examples x 200 neurons): white = saturated, black = active. What you look for:

- **Entire column ■** = **dead neuron** — no example ever activates it in the useful range. It will never learn. Permanent brain damage.
- **Scattered ■** = some saturation but neurons still get gradient from other examples. Recoverable, but wasteful.

### 2.4 Dead Neurons Across Different Nonlinearities

| Nonlinearity   | Dead neuron condition            | Characteristics                                                                                                    |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **tanh**       | Output always ±1                 | Gradient ≈ 0 in tails, but never exactly 0                                                                         |
| **sigmoid**    | Output always 0 or 1             | Same issue as tanh (squashing function)                                                                            |
| **ReLU**       | Pre-activation always < 0        | Gradient is **exactly** 0. Completely dead. Can happen at init or during training (knocked off by large gradient). |
| **Leaky ReLU** | Never fully dead                 | Always has some gradient, even for negative inputs                                                                 |
| **ELU**        | Partially dead for very negative | Has flat region for very negative values                                                                           |

**ReLU dead neurons** are particularly dangerous:

- Can happen at initialization by chance
- Can happen during training if learning rate is too high — a neuron gets "knocked off" the data manifold and never activates again
- This is truly permanent — the neuron is dead forever

### 2.5 The Fix

Scale down W1 so pre-activations stay in the active range of tanh:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * 0.2
b1 = torch.randn(n_hidden) * 0.01  # small entropy, not random
```

```
Before fix (W1 ~ N(0, 1)):           After fix (W1 ~ N(0, 0.2)):

h_preact values: [-12, 8, -15, ...]   h_preact values: [-1.2, 0.8, -1.5, ...]
                    ↓ tanh                                  ↓ tanh
h values:        [-1.0, 1.0, -1.0, ...]  h values:     [-0.83, 0.66, -0.91, ...]
                  all saturated!                          in active range!

Histogram of h:                        Histogram of h:
 ▌              ▌                          ▃▅█▇▅▃
-1       0      1                      -1    0     1
90% saturated → gradients dead         ~0% saturated → gradients flow
```

After this fix, pre-activations range from roughly -1.5 to 1.5 instead of -15 to 15, and saturation drops from ~90% to nearly 0%.

### 2.6 Impact

| Configuration           | Validation Loss |
| ----------------------- | --------------- |
| Original                | 2.17            |
| + Fixed softmax         | 2.13            |
| + Fixed tanh saturation | **2.10**        |

---

## 3. Principled Initialization: Kaiming Init

### 3.1 The Problem with Magic Numbers

We used `* 0.2` and `* 0.01` — where did these come from? Just looking at histograms and guessing. This doesn't scale to large networks with many layers.

### 3.2 The Variance Preservation Principle

If the input `x` is a unit gaussian (mean=0, std=1), we want the output `y = x @ W` to also be roughly unit gaussian.

```python
x = torch.randn(1000, 10)       # unit gaussian input
W = torch.randn(10, 200)        # random weights

y = x @ W
print(y.std())  # ≈ 3.16 — it EXPANDED! Not what we want.
```

```
Without normalization:

x ~ N(0, 1)        W ~ N(0, 1)          y = x @ W
(1000, 10)          (10, 200)            (1000, 200)

std = 1.0           std = 1.0            std ≈ 3.16 ← EXPANDED!
 ▁▃█▇▃▁             ▁▃█▇▃▁               ▁▃█▇▃▁
-3  0  3            -3  0  3           -10   0   10

Each output neuron sums 10 products of gaussians → variance grows by fan_in
```

The standard deviation grows with the fan-in. The fix:

```python
# Divide by sqrt(fan_in) to preserve the standard deviation
W = torch.randn(10, 200) / (10 ** 0.5)

y = x @ W
print(y.std())  # ≈ 1.0 — preserved! ✓
```

```
With normalization (÷ sqrt(fan_in)):

x ~ N(0, 1)        W ~ N(0, 1/√10)     y = x @ W
(1000, 10)          (10, 200)            (1000, 200)

std = 1.0           std ≈ 0.316          std ≈ 1.0 ← PRESERVED! ✓
 ▁▃█▇▃▁               ▃█▃                ▁▃█▇▃▁
-3  0  3            -1  0  1            -3   0   3
```

**Mathematical basis**: When multiplying a unit gaussian input by a weight matrix, the variance of the output scales by `fan_in`. Dividing by `sqrt(fan_in)` compensates exactly.

### 3.3 Kaiming He et al. (2015) — Adding Gain for Nonlinearities

Nonlinearities like tanh and ReLU are **contractive** — they squeeze the distribution. To compensate, we need a **gain** factor:

```
std = gain / sqrt(fan_in)
```

| Nonlinearity  | Gain            | Why                                |
| ------------- | --------------- | ---------------------------------- |
| Linear (none) | 1               | No squashing                       |
| tanh          | 5/3 ≈ 1.667     | Moderate squashing of tails        |
| ReLU          | sqrt(2) ≈ 1.414 | Discards entire negative half      |
| sigmoid       | 1               | (but rarely used in hidden layers) |

```python
# Kaiming init for our tanh network:
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / (n_embd * block_size)**0.5
# gain = 5/3, fan_in = n_embd * block_size = 30
# std ≈ 0.30 (compare to our manual guess of 0.2)
```

```
Computing the std for W1:

  std = gain / sqrt(fan_in)
      = (5/3) / sqrt(30)
      = 1.667 / 5.477
      ≈ 0.304

W1 ~ N(0, 0.304):     After Linear:        After tanh:
                       y = x @ W1           h = tanh(y)

    ▂▅█▅▂                ▂▅█▅▂                ▂▅█▅▂
  -1  0   1            -3  0  3             -1  0  1
  std ≈ 0.30           std ≈ 1.0            std ≈ 0.65

The gain (5/3) compensates for tanh's squashing,
so activations remain well-scaled layer after layer.
```

### 3.4 PyTorch Implementation

```python
# PyTorch provides this directly:
torch.nn.init.kaiming_normal_(W1, mode='fan_in', nonlinearity='tanh')

# mode='fan_in' (default): normalize forward pass activations
# mode='fan_out': normalize backward pass gradients
# The paper finds these are approximately equivalent
```

**PyTorch's default `nn.Linear` init** uses `1 / sqrt(fan_in)` with a uniform distribution (gain=1). It doesn't account for the nonlinearity gain — you'd need to apply that separately.

### 3.5 Impact

With Kaiming init (gain=5/3, std ≈ 0.30) vs manual guess (0.2):

| Configuration           | Validation Loss |
| ----------------------- | --------------- |
| Manual scaling (\* 0.2) | 2.10            |
| Kaiming init (5/3 gain) | **~2.10**       |

Same result — but now it's **principled** and will scale to much deeper networks without manual tuning.

---

## 4. Batch Normalization

### 4.1 The Core Idea

If we want pre-activations to be unit gaussian... **why not just normalize them?**

This sounds almost too simple, but it works. The normalization operation is fully differentiable, so gradients flow through it.

### 4.2 The Formula

From the paper (Ioffe & Szegedy, 2015):

```
1. Calculate batch mean:     μ = (1/m) Σ x_i
2. Calculate batch variance:  σ² = (1/m) Σ (x_i - μ)²
3. Normalize:                 x̂_i = (x_i - μ) / sqrt(σ² + ε)
4. Scale and shift:           y_i = γ * x̂_i + β
```

```python
# Implementation:
# h_preact shape: (batch_size, n_hidden) = (32, 200)

# Step 1-2: compute statistics over the batch dimension
bn_mean = h_preact.mean(dim=0, keepdim=True)   # (1, 200)
bn_std  = h_preact.std(dim=0, keepdim=True)     # (1, 200)

# Step 3: normalize
h_preact = (h_preact - bn_mean) / bn_std

# Step 4: learnable scale and shift
bn_gain = torch.ones(1, n_hidden)    # γ — initialized to 1
bn_bias = torch.zeros(1, n_hidden)   # β — initialized to 0
h_preact = bn_gain * h_preact + bn_bias
```

```
h_preact (32, 200) — before BatchNorm:

          neuron 0    neuron 1    neuron 2   ...  neuron 199
ex  0:      3.2        -1.5         0.8             2.1
ex  1:      2.8        -0.3         1.2             1.5
ex  2:      4.1        -2.1         0.3             3.3
 ...         ...        ...         ...             ...
ex 31:      3.5        -1.0         0.6             2.7
            ─────       ─────       ─────           ─────
 mean:       3.4        -1.2         0.7             2.4    ← bn_mean (1, 200)
 std:        0.5         0.7         0.3             0.6    ← bn_std  (1, 200)

Step 3: Normalize each column independently:
          (value - mean) / std

          neuron 0    neuron 1    neuron 2   ...  neuron 199
ex  0:     -0.4        -0.4         0.3            -0.5
ex  1:     -1.2         1.3         1.7            -1.5
ex  2:      1.4        -1.3        -1.3             1.5
 ...                  ← unit gaussian per column!

Step 4: Scale by γ and shift by β (both learnable, per neuron):
  output = γ * normalized + β
  At init: γ=1, β=0 → output = normalized (unit gaussian)
  After training: γ and β adjust to whatever the network needs
```

### 4.3 Why Gamma and Beta?

At initialization:

- `γ = 1, β = 0` → output is exactly unit gaussian (what we want)

During training:

- Backpropagation adjusts γ and β so the network can learn whatever distribution works best
- Maybe some neurons should be more "trigger happy" (shifted mean)
- Maybe some should have wider spread (larger gamma)

Without γ and β, the layer would be **forced** to always output unit gaussian — too restrictive.

### 4.4 Training vs Inference

**Problem**: BatchNorm uses batch statistics (mean, variance). But at inference time, you might feed a single example — there's no "batch" to compute statistics from.

**Solution**: Maintain running estimates during training, use them at inference.

```python
# Initialize buffers
bn_mean_running = torch.zeros(1, n_hidden)
bn_std_running  = torch.ones(1, n_hidden)

# During training, after computing bn_mean and bn_std for the current batch:
with torch.no_grad():
    bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_mean
    bn_std_running  = 0.999 * bn_std_running  + 0.001 * bn_std

# At inference time, use the running estimates instead of batch stats:
h_preact = (h_preact - bn_mean_running) / bn_std_running
h_preact = bn_gain * h_preact + bn_bias
```

```
Running mean update over training steps (momentum = 0.001):

Step 0:  running = 0.000                    (initialized)
Step 1:  running = 0.999 × 0.000 + 0.001 × 3.42 = 0.003
Step 2:  running = 0.999 × 0.003 + 0.001 × 3.51 = 0.007
 ...                  (slowly accumulates)
Step 1000: running ≈ 2.18                   (converging)
 ...
Step 10000: running ≈ 3.40                  (close to true mean)

After training → running ≈ true mean over entire training set
                 → use this fixed value at inference (no batch needed!)
```

The momentum (0.001) controls the update speed:

- **Large batch size** → batch statistics are stable → can use higher momentum (e.g., 0.1, PyTorch default)
- **Small batch size** (e.g., 32) → batch statistics are noisy → use lower momentum (e.g., 0.001) so running estimate doesn't thrash

**Alternative**: Skip the running estimate entirely, calibrate once after training:

```python
# Post-training calibration (stage 2)
with torch.no_grad():
    # Forward pass entire training set
    emb = C[X_train]
    h_preact = emb.view(-1, n_embd * block_size) @ W1 + b1
    bn_mean = h_preact.mean(dim=0, keepdim=True)
    bn_std  = h_preact.std(dim=0, keepdim=True)
    # Use these fixed values at inference
```

But nobody wants a second stage, so the running estimate is preferred.

### 4.5 Important Details

**Epsilon**: A tiny constant (default 1e-5) added inside the square root to prevent division by zero:

```python
h_preact = (h_preact - bn_mean) / torch.sqrt(bn_var + 1e-5)
```

**Bias in preceding layer is useless**: If the layer before BatchNorm has a bias `b1`:

```python
h_preact = emb @ W1 + b1  # b1 gets added...
h_preact = (h_preact - bn_mean) / bn_std  # ...then immediately subtracted out by mean
```

```
Example for one neuron across a batch:

emb @ W1 column:   [2.0, 3.0, 1.0, 4.0]     (4 examples)
+ b1 = 5.0:        [7.0, 8.0, 6.0, 9.0]     bias shifts everything up
mean = 7.5:        [7.0, 8.0, 6.0, 9.0] - 7.5 = [-0.5, 0.5, -1.5, 1.5]

Without b1:         [2.0, 3.0, 1.0, 4.0]
mean = 2.5:        [2.0, 3.0, 1.0, 4.0] - 2.5 = [-0.5, 0.5, -1.5, 1.5]
                                                    ↑ SAME RESULT!
The bias just shifts the mean, then BN subtracts the mean → bias is useless.
```

Whatever `b1` adds, the mean subtraction removes. So `b1.grad` will be zero — it never learns. Remove it:

```python
# Don't use bias in layers before BatchNorm
W1 = torch.randn((fan_in, n_hidden)) * std  # no b1
# BatchNorm's own beta (bn_bias) handles the biasing instead
```

This is why in PyTorch you see `nn.Conv2d(..., bias=False)` before `nn.BatchNorm2d()`.

### 4.6 The Coupling Problem

BatchNorm introduces a deeply strange property: **examples in a batch are mathematically coupled**.

Before BatchNorm: each example is processed independently. The activations for input X depend only on X.

After BatchNorm: the activations for input X depend on **what other examples happen to be in the batch**. The mean and std are computed over the batch, so changing one example changes the normalization for all others.

**Surprising benefit**: This coupling acts as a **regularizer**. It introduces noise/jitter into each example's activations (because the batch statistics change each iteration), which is a form of data augmentation. This makes it harder to overfit.

**The cost**: BatchNorm causes a huge number of bugs. The coupling between examples leads to subtle, hard-to-debug issues. Everyone dislikes this property.

**Modern alternatives** that don't couple examples:

- **Layer Normalization** — normalizes across features within each example
- **Group Normalization** — normalizes across groups of features within each example
- **Instance Normalization** — normalizes each feature map independently

These have become more common in recent architectures (especially Transformers use LayerNorm), but BatchNorm persists because it works well, partly due to the regularization effect.

### 4.7 Where to Place BatchNorm

The standard motif in deep networks:

```
Linear/Conv → BatchNorm → Nonlinearity (ReLU/tanh)
```

Repeated for each layer. You can also place it after the nonlinearity — results are similar.

### 4.8 BatchNorm Parameters Summary

| Component        | Learned?    | How?                       |
| ---------------- | ----------- | -------------------------- |
| γ (gain/scale)   | Yes         | Backpropagation            |
| β (bias/shift)   | Yes         | Backpropagation            |
| Running mean     | No (buffer) | Exponential moving average |
| Running variance | No (buffer) | Exponential moving average |

---

## 5. Diagnostic Tools for Training

### 5.1 Activation Histograms (Forward Pass)

For each tanh layer, plot the histogram of output values:

```python
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        mean = t.mean().item()
        std  = t.std().item()
        saturated = (t.abs() > 0.97).float().mean().item()
        print(f'Layer {i}: mean={mean:.4f}, std={std:.4f}, saturated={saturated:.4f}')
```

**What to look for**:

- Standard deviation should be **roughly equal** across layers (~0.65 for tanh)
- Saturation should be **~5%** — some is fine, too much means vanishing gradients
- Std shrinking toward zero across layers → gain too low
- Std growing across layers → gain too high

### 5.2 Gradient Histograms (Backward Pass)

```python
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        mean = t.mean().item()
        std  = t.std().item()
        print(f'Layer {i}: mean={mean:.4f}, std={std:.4f}')
```

**What to look for**:

- Std should be **roughly equal** across layers
- Std shrinking toward deeper layers → vanishing gradients
- Std growing toward deeper layers → exploding gradients

### 5.3 Update-to-Data Ratio (The Key Metric)

This is the most important diagnostic. It measures how much each parameter actually changes per step relative to its current magnitude:

```python
# Track over time during training
ud = []  # update-to-data ratios

for i in range(num_steps):
    # ... forward, backward ...

    with torch.no_grad():
        ratios = []
        for p in parameters:
            update = lr * p.grad
            ratio = (update.std() / p.data.std()).log10().item()
            ratios.append(ratio)
        ud.append(ratios)

    # ... parameter update ...

# Plot: x-axis = training step, y-axis = log10(update/data) for each parameter
```

**The rule of thumb**:

```
log10(update_std / data_std) ≈ -3
```

| Ratio (log10)    | Meaning                                                   |
| ---------------- | --------------------------------------------------------- |
| ~ -3             | Good. Updates are ~1/1000 of the parameter magnitude.     |
| >> -3 (e.g., -1) | **Learning rate too high.** Parameters changing too fast. |
| << -3 (e.g., -5) | **Learning rate too low.** Parameters barely changing.    |

```
Update-to-data ratio
  |
  |  -1 --------  ← too high (LR too large)
  |
  |  -2 --------
  |
  |  -3 ========  ← target zone
  |
  |  -4 --------
  |
  |  -5 --------  ← too low (LR too small)
  +-------------------> training step
```

### 5.4 Gradient-to-Data Ratio

A simpler version — just compare gradient magnitude to parameter magnitude:

```python
for p in parameters:
    if p.ndim == 2:  # weights only, skip biases
        ratio = (p.grad.std() / p.data.std()).item()
        print(f'{p.shape}: grad/data = {ratio:.4f}')
```

Watch for outlier layers where this ratio is 10x or 100x different from others — they'll train at very different speeds with a single global learning rate.

---

## 6. Principled Code: PyTorch-Style Modules

### 6.1 Linear Layer Module

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5  # Kaiming
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

Mirrors `torch.nn.Linear(fan_in, fan_out, bias=True)`.

### 6.2 BatchNorm1d Module

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Learnable parameters
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        # Buffers (not learned by gradient descent)
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar  = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

Mirrors `torch.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1)`.

### 6.3 Tanh Module

```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

### 6.4 Stacking into a Network

```python
# Build the network as a list of layers
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),                       BatchNorm1d(vocab_size),
]

# Collect all parameters
parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

# Forward pass: sequential application
x = C[X_batch]
x = x.view(-1, n_embd * block_size)
for layer in layers:
    x = layer(x)
logits = x
loss = F.cross_entropy(logits, Y_batch)
```

```
Shape flow through the network (batch_size=32, n_hidden=200, vocab_size=27):

 C[X_batch]                        → (32, 3, 10)
 .view(-1, 30)                     → (32, 30)        ← flatten embeddings
   ↓
 Linear(30, 200, bias=False)       → (32, 200)       ← W: (30, 200)
 BatchNorm1d(200)                  → (32, 200)       ← normalize each neuron
 Tanh()                            → (32, 200)       ← squash to [-1, 1]
   ↓
 Linear(200, 200, bias=False)      → (32, 200)       ← W: (200, 200)
 BatchNorm1d(200)                  → (32, 200)
 Tanh()                            → (32, 200)
   ↓
 ... (repeat 3 more times) ...     → (32, 200)
   ↓
 Linear(200, 27)                   → (32, 27)        ← W: (200, 27)
 BatchNorm1d(27)                   → (32, 27)        ← final logits
   ↓
 F.cross_entropy(logits, Y)        → scalar           ← loss
```

Note the pattern: `Linear(bias=False)` → `BatchNorm1d` → `Tanh`, repeated. The last layer has `BatchNorm1d` but no `Tanh` (the softmax in cross-entropy handles the output).

---

## 7. Real-World Example: ResNet

The exact same pattern appears in production networks. A ResNet bottleneck block:

```python
# From torchvision ResNet implementation
class Bottleneck(nn.Module):
    def __init__(self, ...):
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)  # bias=False!
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)  # bias=False!
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, 1, bias=False)  # bias=False!
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # Conv → BN → ReLU
        out = self.relu(self.bn2(self.conv2(out)))  # Conv → BN → ReLU
        out = self.bn3(self.conv3(out))             # Conv → BN
        out += identity                              # Residual connection
        out = self.relu(out)
        return out
```

Key observations:

- **`bias=False`** on all conv layers — because BatchNorm follows each one
- The motif **Conv → BatchNorm → ReLU** is repeated exactly
- Convolutions are just spatial versions of linear layers (W\*x + b on overlapping patches)
- This is exactly what the lecture teaches, applied to real image classification

---

## 8. Why Nonlinearities Are Necessary

Without nonlinearities, a stack of linear layers collapses to a single linear transformation:

```
y = W3(W2(W1 * x + b1) + b2) + b3
  = (W3 * W2 * W1) * x + (combined bias)
  = W_effective * x + b_effective  ← just one linear layer!

Example with 2D weights:

W1 = [2, 0]    W2 = [1, 3]    W1 × W2 = [2, 6]
     [0, 1]         [0, 1]              [0, 1]

Two layers collapse into one equivalent matrix.
No matter how many layers: still just a linear transform.

With tanh between layers:
  W1 → tanh → W2 → tanh → W3
  Now it's NOT collapsible — each tanh bends the space
  → can approximate any function (universal approximation)
```

No matter how many linear layers you stack, the result is always a linear function of the input.

**With nonlinearities** (tanh, ReLU, etc.), the network can approximate **any arbitrary function** (universal approximation theorem). The nonlinearity between layers is what gives depth its power.

**Interesting aside**: Even though a linear sandwich collapses to one linear transformation in the forward pass, the **training dynamics** (backward pass, gradient flow) are different. There's active research on infinitely deep linear networks and their optimization properties.

---

## 9. Key Takeaways

### Progression of Fixes and Their Impact

| Fix                                             | Val Loss | Improvement                                   |
| ----------------------------------------------- | -------- | --------------------------------------------- |
| Original MLP (Part 2)                           | 2.17     | baseline                                      |
| + Fixed softmax confidence (W2 \* 0.01, b2 = 0) | 2.13     | -0.04                                         |
| + Fixed tanh saturation (W1 \* 0.2)             | 2.10     | -0.03                                         |
| + Kaiming init (principled, same effect)        | ~2.10    | same, but principled                          |
| + Batch Normalization                           | ~2.10    | same for shallow net, essential for deep nets |

### Core Lessons

- **Know your expected initial loss.** For classification with N classes: `-log(1/N)`. Anything much higher means broken initialization.
- **Logits should be ~zero at init.** Scale down the last layer weights and zero the bias.
- **Pre-activations should be in the active zone of their nonlinearity.** Not too large (saturation), not too small (inactive).
- **The vanishing gradient problem is real.** Saturated tanh/sigmoid neurons kill gradient flow. Dead ReLU neurons are permanently lost.
- **Use Kaiming init**: `std = gain / sqrt(fan_in)` with appropriate gain for your nonlinearity.
- **Batch Normalization stabilizes deep networks** by explicitly normalizing activations to be unit gaussian (then allowing learned scale/shift).
- **Monitor the update-to-data ratio.** Target ~1e-3 on log10 scale. This catches bad learning rates and miscalibrated layers.
- **`bias=False` before BatchNorm.** The bias is redundant since BatchNorm subtracts the mean.

### Initialization & Diagnosis Checklist

- [ ] Verify initial loss matches `-log(1/N)` for your number of classes
- [ ] Scale last layer weights small, biases to zero
- [ ] Use Kaiming initialization for all weight matrices
- [ ] Add BatchNorm layers after linear/conv layers
- [ ] Remove bias in layers immediately before BatchNorm
- [ ] Plot activation histograms: check for tanh saturation (~5% is okay)
- [ ] Plot gradient histograms: check for symmetry across layers
- [ ] Track update-to-data ratio over training: should hover around -3 on log10 scale
- [ ] If ratio >> -3: reduce learning rate
- [ ] If ratio << -3: increase learning rate

### Historical Context

Before modern innovations (BatchNorm, residual connections, Adam optimizer), training deep networks was like **balancing a pencil on your finger** — every weight scale had to be precisely calibrated. These innovations have made it much more forgiving, but understanding the underlying principles is still essential for:

- Debugging when things go wrong
- Working with novel architectures
- Pushing performance in competitive settings

---

## 10. Looking Forward

The current bottleneck is **not optimization** (BatchNorm and proper init handle that) — it's the **context length**. We're only using 3 characters to predict the next one. To improve further:

- **Recurrent Neural Networks (RNNs)** — process sequences of arbitrary length
- RNNs are essentially **very deep networks** when unrolled through time
- Everything from this lecture (activation statistics, gradient flow, normalization) becomes **critical** for training RNNs
- Variants like **GRU** and **LSTM** were invented specifically to address the vanishing gradient problem in deep/recurrent networks

---

_Notes based on Andrej Karpathy's makemore lecture series, Part 3._
