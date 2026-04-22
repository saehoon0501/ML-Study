# Study Log — Weak Points & Review Notes

Track what you got wrong, why, and whether you've fixed it.
Review this file before starting each new phase.

---

## Phase 1: Neural Network Foundations

### Day 1-7 Exercise Results (2026-04-20)

| Topic | Issue | Severity | Status |
|-------|-------|----------|--------|
| Backprop (A1) | Computed symbolic derivatives but didn't plug in final values. Left `df/da = -6e` instead of computing `12`. | Minor | [ ] Fixed |
| Backprop (A3) | Missed the real bug: **gradient zeroing**. Identified "loss averaging" which is valid but secondary. Confused the symptom — it causes divergence, not jumping. | Medium | [ ] Fixed |
| Micrograd (B1) | Correct local gradients but forgot to multiply by `out.grad`. Writing `self.grad += 1` instead of `self.grad += 1.0 * out.grad` breaks the chain rule — you only get local derivatives, not end-to-end. | High | [ ] Fixed |
| Softmax (C4) | Said high confidence "needs regularization" — not necessarily. High confidence + correct = ideal. High confidence + wrong = problem. Uniform confidence = untrained model, loss = `-log(1/n_classes)`. | Minor | [ ] Fixed |
| Bigrams (D1) | Gave raw counts instead of probabilities. Forgot to normalize (divide by row sums). | Medium | [ ] Fixed |
| Bigrams (D2) | Used raw counts in the NLL formula instead of probabilities. Got `-(log 2)/3` instead of correct `0.828`. | Medium | [ ] Fixed |
| Bigrams (D4) | Left blank. Key concept: one-hot @ W = row selection (table lookup). W converges to `log(P)` because `softmax(log(P)) = P`. | High | [ ] Fixed |
| Exercises E1-E3 | Completed 2026-04-20. See below. | — | [x] Done |

### E1-E3 Results (2026-04-20)

| Topic | Issue | Severity | Status |
|-------|-------|----------|--------|
| E1: Overfitting | Didn't identify overfitting as the cause of "low training loss + garbage output." Gave vague answer about distribution design. Key: always track **validation loss** alongside training loss. | High | [ ] Fixed |
| E2: Expected initial loss | Didn't recognize that 3.29 ≈ `-log(1/27)` = uniform prediction = correct starting point. This is a fundamental sanity check — if initial loss deviates from `-log(1/n_classes)`, something is broken. | High | [ ] Fixed |
| E3-3: ReLU vs sigmoid | Wrong reason. Not about "ignoring negatives." The real reason: **vanishing gradients.** Sigmoid gradient maxes at 0.25 → multiplied across layers → gradients shrink exponentially → early layers can't learn. ReLU gradient is 0 or 1 — gradients pass through unchanged. | High | [ ] Fixed |

### Patterns to Watch

1. **Precision gap:** You understand concepts but skip mechanical steps (normalizing, plugging in values). This will cause shape mismatches and silent bugs in PyTorch code.
2. **`out.grad` omission:** The chain rule requires multiplying by the upstream gradient. Every `_backward()` function must include `* out.grad`. This is not optional — it IS backpropagation.
3. **Counts vs probabilities:** Always normalize before computing loss. Raw counts are not probabilities.
4. **Overfitting awareness:** When training loss drops but quality is bad, the first suspect is overfitting. Always have a validation set. Training loss alone tells you nothing about generalization.
5. **Initial loss sanity check:** Before training, loss should equal `-log(1/n_classes)`. Memorize this — it's a one-second health check that catches initialization bugs, data leakage, and broken shuffling.
6. **Vanishing gradients:** Understand WHY architectural choices exist, not just what they do. ReLU, residual connections, layer norm — all exist to keep gradients flowing through deep networks.

---

### Karpathy Video 5: Let's reproduce GPT-2 (2026-04-22)

Questions asked while working through `build-nanogpt`. Each row records the confusion, the correct concept, and the takeaway.

| Topic | Question / Confusion | Clarified Concept | Status |
|-------|---------------------|-------------------|--------|
| Cloud GPU choice | Which rental option fits the video's training run? | Karpathy uses **Lambda Labs 8× A100 40GB**, ~90 min for full 10B-token run (~$15–20). For learning: step through on Colab free → $3 shortened run on 1× A100 → $15 full reproduction. | [ ] Fixed |
| `FileNotFoundError: edu_fineweb10B` | Thought error was about loading pretrained GPT-2 weights. | Error is about **missing training data**. `train_gpt2.py` trains from scratch (line 339); the `from_pretrained` path (line 340) is commented out. Must run `fineweb.py` first to download + tokenize FineWeb-Edu shards. | [ ] Fixed |
| Data loader `+1` | Thought `tokens[:24+1]` with `buf[1:]` was "skipping index 0." | It's **next-token prediction shift**. `buf` has 25 tokens so `x = buf[:-1]` (24) and `y = buf[1:]` (24) form input/target pairs where `y[i]` is the token after `x[i]`. Nothing is skipped — index 0 is still in `x`. | [ ] Fixed |
| `(B, T, C)` notation | What does T stand for? | **T = time** (sequence length / context length). Convention: B = batch, T = time, C = channels (embedding dim). Each token position is a "timestep." | [ ] Fixed |
| B and T vs file | Thought B = number of sentences, T = size of file. | Both are **hyperparameters you choose**, independent of file. B = how many sequences run in parallel per step. T = context length (GPT-2 uses 1024). GPT-2 training has no concept of sentences — data is one token stream with `<\|endoftext\|>` as document separators. File size only determines steps-per-epoch. | [ ] Fixed |
| `nn.init.normal_` vs `torch.randn` | What's `std` in `nn.init.normal_`, how does it relate to `torch.randn`? | `nn.init.normal_(w, 0, std)` fills in-place from `N(0, std²)`. `torch.randn(shape)` samples raw `N(0, 1)` — multiply by `std` for the same effect. Use `nn.init.*` for params, `torch.randn` for data/noise tensors. | [ ] Fixed |
| `std=0.02` | Why this specific number? | GPT-2 paper's **empirical default**. Close to but smaller than Xavier (~0.036 for `n_embd=768`). Not derived from theory. | [ ] Fixed |
| `std *= (2 * n_layer) ** -0.5` | Meaning of `**-0.5` and why divide by `sqrt(2 * n_layer)`? | `** -0.5` = `1/sqrt(x)`. This is the **residual-stream variance fix**: each block adds 2 contributions (attn + mlp) to the stream, so `2 * n_layer` total. Scaling each residual branch's output projection by `1/sqrt(2 * n_layer)` keeps the residual stream's variance stable at init. Without it, deeper networks blow up. | [ ] Fixed |
| Which layers get scaled | Thought scaling applied to "layers with multiple residual paths." | Scaling applies to the **output projection of each residual branch** — i.e., the last `Linear` whose output is directly added to the residual stream. Specifically `c_proj` in both attention and MLP (marked `NANOGPT_SCALE_INIT=1`). Earlier layers like `c_attn` / `c_fc` use default `std=0.02`. | [ ] Fixed |
| INT vs FP: training vs inference | Why does INT8/INT4 work at inference but break training? | Inference = one-shot forward pass; quantization noise doesn't compound. Training needs to represent **tiny gradients (~1e-5)** and **tinier updates (~1e-9)** — INT8's smallest step is 1, so gradients round to zero. Errors also compound across millions of steps. Production training uses **mixed precision**: BF16/FP16 for matmuls, FP32 master weights + FP32 optimizer state (Adam `v`, `m`). FP8 training is now viable (H100); INT4 training is research-only. | [ ] Fixed |
| Weight decay scope | Why decay only matrix (2D) params, not biases / LayerNorm scales? | Weight decay assumes "smaller = more regularized." That holds for matrices because they are **scale-invariant when followed by normalization**: `normalize(c·W·x) = normalize(W·x)`, so the norm of W is a redundant degree of freedom that drifts large and shrinks the effective LR — decay controls it. **Biases and norm scales are not scale-invariant** — their scalar values have direct meaning. Decaying biases can kill ReLU neurons (push into dead zone); decaying LayerNorm γ toward 0 suffocates the layer (`y → β`, constant). Standard recipe: split params by `p.dim() >= 2` into decay / no-decay groups in AdamW. | [ ] Fixed |

### Patterns to Watch (GPT-2 video)

1. **Read the code, don't guess at errors.** The `edu_fineweb10B` error looked like a weight-loading issue but was a data-prep issue. Always check which code path is active (trained-from-scratch vs `from_pretrained`).
2. **Next-token shift is the foundation.** `x = buf[:-1]`, `y = buf[1:]` is the pattern for causal LM training. Internalize this — you'll see it in every LM dataloader.
3. **B and T are compute choices, not data properties.** B is for GPU parallelism; T is for model context. Neither is "how much data you have."
4. **Init math matters.** `(2 * n_layer) ** -0.5` isn't a magic number — it's the variance-preservation factor for residual streams. Every deep architecture has an analogous trick.
5. **Precision hierarchy:** INT for inference (frozen, one-shot), FP for training (gradients need dynamic range), FP32 master weights for optimizer state (small updates must not vanish).

---

<!-- Future phases will be appended below -->
