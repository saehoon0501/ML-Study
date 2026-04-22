# Training Speed Optimization — Symptom Checklist

A diagnostic reference for speeding up neural network training. Each section starts with an observable symptom; work top-down through its checklist (cheap/high-impact items first).

**How to read citations:** Each checklist item ends with reference tags like `[1][7]` pointing to the References section at the bottom. Use them to verify claims before applying a change to a production run.

**Revision note (2026-04-22):** Several recommendations updated based on 2024–2025 research. Items marked ⚠ carry recently-discovered caveats; items marked 🆕 are 2024+ developments that supersede older advice in Karpathy's 2024 video.

---

## Quick baseline: always-on optimizations

Before debugging any symptom, confirm these are enabled. Most are one-line changes that cost nothing.

- [ ] `torch.set_float32_matmul_precision('high')` — enables TF32 on Ampere+ (A100/H100). **~2.5× matmul speedup** at FP32 with negligible accuracy loss. [1][9][29]
- [ ] ⚠ Wrap forward in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` — **~2× speedup**, minimal accuracy loss. Prefer BF16 over FP16 on Ampere+ (no loss scaling needed). **Caveat:** BF16 + Flash Attention can diverge due to biased rounding errors when attention probabilities saturate at 1.0 — see "Loss diverges" section. Always baseline against TF32 for critical runs. [1][3][7][30]
- [ ] ⚠ Use `F.scaled_dot_product_attention` (SDPA) instead of manual `softmax(QK^T/√d)V`. Automatically dispatches to **Flash Attention** when shapes support it. **Caveat:** see BF16 interaction above. [1][4][5][8][30]
- [ ] `model = torch.compile(model)` — kernel fusion + Python overhead removal. **~1.5–2× speedup** on A100-class hardware after warmup; **consumer GPUs (T4, older) see only ~5–10%**. Verify with `TORCH_LOGS="+dynamo"` that compilation isn't silently falling back. [1][6][29]
- [ ] Use `torch.optim.AdamW(..., fused=True)` — single-kernel optimizer step instead of dozens of small ones. [1]
- [ ] Pad vocab / hidden dims to **multiples of 64 or 128**. Example: GPT-2's `vocab_size=50257` → pad to `50304` (next multiple of 128). Tensor cores require aligned dims for peak throughput. [1][9]
- [ ] Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — prevents loss spikes that waste steps. GPT-3 recipe. [1][21]
- [ ] Weight decay **only on 2D params** (matrices), not on biases or LayerNorm scales. Standard in GPT-2/GPT-3. [1][21]
- [ ] 🆕 **LR schedule: prefer WSD (Warmup-Stable-Decay) or linear decay-to-zero over cosine.** 2024–2025 research shows linear decay-to-zero outperforms cosine at scale. Modern warmup windows are short: 1–2% of total steps (not GPT-3's 375M-token warmup). Cosine still works; it's just no longer SOTA. [31][32]

---

## Symptom: GPU utilization is low (< 70%)

The GPU is idle waiting for something else — usually data loading or Python overhead.

Check with: `nvidia-smi dmon -s u` (watch `sm` column) or PyTorch profiler. [27]

- [ ] **Check data loading.** Is `DataLoader(num_workers=...)` set? Start with `num_workers = 4 * num_gpus`. [10]
- [ ] Enable `pin_memory=True` in DataLoader for faster CPU→GPU transfer (uses page-locked memory). [10]
- [ ] Set `persistent_workers=True` — avoids respawning worker processes every epoch. [10]
- [ ] Add `prefetch_factor=4` (or higher) so batches are ready before GPU asks. [10]
- [ ] Is the dataset on fast storage? Move to local NVMe if reading from network disk.
- [ ] Pre-tokenize / pre-process data offline. Never tokenize inside the training loop. [1][2]
- [ ] Reduce Python overhead in the hot loop: no `.item()`, no `.cpu()`, no `print()` per step. [1]
- [ ] Are you logging loss every step? `loss.item()` forces a CPU↔GPU sync. Log every N steps or accumulate on GPU. [1]
- [ ] Use `torch.compile(model)` — eliminates per-op Python dispatch overhead. [6]
- [ ] If using custom kernels, check they're not falling back to CPU.

---

## Symptom: GPU is at 100% but training is still slow

GPU is busy but not doing useful work efficiently. Time to tune kernels and precision.

- [ ] Enable **mixed precision** (BF16 on Ampere+, FP16 otherwise). Biggest single lever. [3][7]
- [ ] Use **Flash Attention** via `F.scaled_dot_product_attention` — O(T²) compute but O(T) memory; especially helps at long context lengths. [4][5][8]
- [ ] `torch.compile(model)` — enables kernel fusion (e.g., fusing GELU + linear). [6]
- [ ] Pad all tensor dimensions to multiples of 64 or 128 (vocab, hidden, heads). [9]
- [ ] Use `fused=True` for optimizer (`AdamW`, `SGD`). [1]
- [ ] Increase batch size — small batches underutilize tensor cores. Aim for matmul dims that fill the GPU. [9]
- [ ] Profile with `torch.profiler` to find the slowest op. Look for unexpected ops (extra `.contiguous()` calls, hidden copies). [27]
- [ ] Check for CPU↔GPU transfers in the hot loop. Every `.to(device)` on a non-tensor costs.
- [ ] ⚠ On H100: **FP8 training** via NVIDIA TransformerEngine offers up to ~2× over BF16, but 2024 research shows significant stability/convergence trade-offs — FP8 narrows the viable hyperparameter space, can fail to converge on noisy data, and requires careful per-tensor scaling. Validate against a BF16 baseline. Not a simple drop-in. [24][33]
- [ ] Check attention mask — a dense `(T, T)` mask is fine, but recomputing it each step wastes time.

---

## Symptom: Out of memory (OOM)

Model, activations, gradients, or optimizer state don't fit.

**Memory budget (BF16 training with Adam):** weights 2B + gradients 2B + optimizer state 8B (Adam `m, v` in FP32) + activations (scales with batch × seq_len × layers). Per param: **~14 bytes** minimum. [14]

- [ ] **Reduce batch size.** Use gradient accumulation to recover effective batch size. [1]
- [ ] Enable **gradient checkpointing** — recomputes activations in backward. Trades ~30% extra compute for massive activation memory savings (often enables 2–4× larger batches). Use `torch.utils.checkpoint` on single-GPU/DDP; for **FSDP, use `torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing`** — raw `torch.utils.checkpoint` conflicts with FSDP's parameter sharding. [15][16][34]
- [ ] Reduce sequence length if possible.
- [ ] Use mixed precision (BF16) — halves weight + activation memory. [3][7]
- [ ] Use a **memory-efficient optimizer** — these are for **memory savings, not speed**. 2024 research shows most modern optimizers perform comparably on final loss when tuned; pick by memory budget: [35]
    - 8-bit AdamW via `bitsandbytes` — nearly matches FP32 AdamW at 1/4 the optimizer-state memory. [17]
    - Adafactor — factorizes second-moment matrix; used in T5. ~50% less state than Adam. [18]
    - Lion — tracks only `m`, half the state of AdamW. Needs 3–10× smaller LR than AdamW; retune. [19]
- [ ] Shard optimizer state across GPUs with **FSDP** or **DeepSpeed ZeRO-2** — splits Adam's `m, v` across ranks. [13][14]
- [ ] For very large models: **ZeRO-3 / FSDP full sharding** — also shards params and grads. [13][14]
- [ ] Offload optimizer state to CPU (ZeRO-Offload). Slow but fits huge models. [14]
- [ ] Check for memory leaks: does memory grow over steps? Look for accumulating tensors in Python lists.
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. [28]
- [ ] Clear grads with `optimizer.zero_grad(set_to_none=True)` (default in newer PyTorch) — frees grad tensors instead of zeroing. [10]
- [ ] Call `.detach()` on anything you save for logging (otherwise you retain the full graph).

---

## Symptom: Can't fit the batch size I want

Two paths: simulate it or distribute it.

- [ ] **Gradient accumulation**: run N micro-batches, call `.backward()` on each, step optimizer every N. Equivalent to `N × micro_batch` batch size. Used in nanoGPT to simulate GPT-3 batch size on a single node. [1]
  ```python
  for i, batch in enumerate(loader):
      loss = model(batch) / accum_steps
      loss.backward()
      if (i + 1) % accum_steps == 0:
          optimizer.step(); optimizer.zero_grad()
  ```
- [ ] **DDP (DistributedDataParallel)**: replicate model on each GPU, split batch across ranks. Near-linear scaling up to ~8 GPUs. [11][12]
- [ ] **FSDP**: shard model across GPUs — fits models larger than a single GPU. [13][14]
- [ ] Combine: DDP + grad accumulation + checkpointing stacks cleanly.
- [ ] Don't forget to scale learning rate with batch size — linear scaling rule up to ~4k batch, then sqrt. [20]

---

## Symptom: Loss diverges / NaN during training

Almost always a numerical precision or init issue, not a speed issue — but you can't speed up a crashed run.

- [ ] **Check initial loss.** Should be `-ln(1/vocab_size)` for LM (e.g., ~10.82 for GPT-2). Wrong initial loss = broken init or data. [1]
- [ ] Enable **gradient clipping** (`clip_grad_norm_` at 1.0). Single most common missing piece. [1][21]
- [ ] Warm up learning rate — start at 0, linearly ramp to target over first ~1% of steps. Prevents early-step blowup. [21][32]
- [ ] Check you're using **BF16, not FP16**. FP16 needs dynamic loss scaling; BF16 doesn't. [3][7]
- [ ] ⚠ **BF16 + Flash Attention divergence.** If loss diverges specifically when you enable SDPA + BF16 but is stable under TF32, this is a known 2024 finding: biased rounding errors when attention probabilities saturate at 1.0 cause systematic gradient corruption. Mitigations: tight gradient clipping (1.0), aggressive warmup, or fall back to TF32 / FP32 attention with BF16 MLP. [30]
- [ ] Check residual stream scaling in init (`std *= (2 * n_layer) ** -0.5` for output projections). [1]
- [ ] Check for `inf`/`nan` in inputs — data bug upstream.
- [ ] Keep optimizer state in **FP32** even with mixed precision. Never quantize Adam's `m, v`. [3]
- [ ] If using FP8: enable "delayed scaling", use per-tensor scaling factors, and plan for a parallel BF16 run to verify convergence. Fall back to BF16 for layers that show instability. [24][33]
- [ ] Consider sequence-length warmup (start training with short sequences, extend) for very deep networks.

---

## Symptom: Multi-GPU scaling is sublinear

Adding more GPUs gives less-than-proportional speedup. You're communication-bound.

- [ ] **Are you using DDP?** Don't use `DataParallel` — it's single-process and slow. [11][12]
- [ ] Use NCCL backend (`init_process_group(backend='nccl')`). Never use Gloo for GPU training. [25]
- [ ] Check interconnect: NVLink (600+ GB/s) vs PCIe (32 GB/s) — massive difference for gradient allreduce. [25]
- [ ] Enable **gradient bucketing**: DDP overlaps allreduce with backward pass automatically, but tune `bucket_cap_mb`. [11][12]
- [ ] Use `find_unused_parameters=False` (the default) unless you truly have unused params. [11]
- [ ] Reduce gradient communication: `gradient_as_bucket_view=True` in DDP. [11]
- [ ] For >8 GPUs: switch to **FSDP** — more communication per step but enables bigger models. [13]
- [ ] For tensor-parallel (Megatron-style): requires NVLink to be viable.
- [ ] Check topology: `nvidia-smi topo -m`. You want `NV#` connections, not `PHB`/`SYS`.

---

## Symptom: First few iterations are slow, then fast

Expected if using `torch.compile` — compilation happens lazily on first call. [6]

- [ ] Confirm: Is the speedup eventually visible after ~5–10 iterations?
- [ ] If not: compile might be falling back silently. Set `TORCH_LOGS="+dynamo"` to see. [6]
- [ ] Use `torch.compile(model, mode="reduce-overhead")` for small models, `"max-autotune"` for large. [6]
- [ ] If recompiling every step: input shapes are changing. Pad to a fixed shape, or use `dynamic=True`. [6]

---

## Symptom: Training gets slower over time

Memory fragmentation, leaks, or cache issues.

- [ ] Is GPU memory usage growing? Look for Python references you're retaining (lists of losses, activations).
- [ ] Clear cache periodically: `torch.cuda.empty_cache()` — but this is usually unnecessary with modern PyTorch.
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. [28]
- [ ] Check that `optimizer.zero_grad(set_to_none=True)` is set. [10]
- [ ] Detach any tensors you store: `losses.append(loss.detach().item())`.
- [ ] Restart periodically for very long runs — state can accumulate.

---

## Symptom: Training on Mac (MPS) is slow vs CUDA

Some features aren't available on MPS backend.

- [ ] MPS doesn't support `bfloat16` autocast reliably — use fp32 or fp16.
- [ ] `torch.compile` support on MPS is spotty — skip it. [6]
- [ ] Flash Attention via SDPA — falls back to math backend on MPS, slower but works. [8]
- [ ] DDP doesn't work on MPS (no NCCL equivalent) — single-device only. [25]
- [ ] Unified memory means batch size is less constrained, but **bandwidth is the bottleneck**: 800 GB/s on M2/M3 Ultra vs 1–3 TB/s on A100/H100.
- [ ] **Bottom line:** Macs are fine for stepping through code, tiny-scale sanity checks, and **inference**. For real training, rent CUDA.

---

## Symptom: Long context is slow / OOM

Attention is O(T²). Memory and compute blow up quadratically in sequence length.

- [ ] Use **Flash Attention** via SDPA — O(T²) compute but **O(T) memory**. Huge win at long context (this is the whole point of Flash Attention). [4][5][8]
- [ ] **Gradient checkpointing** for attention layers — recomputes attention in backward. [15][16]
- [ ] Consider **sliding window attention** (Mistral-style) — O(T × W) instead of O(T²). [22]
- [ ] 🆕 For very long contexts, consider **hybrid Mamba-Attention architectures** — Samba (2024) trains on 4K and extrapolates to 256K+, Jamba combines Mamba with MoE and sliding-window attention. Linear memory in sequence length. [23][36]
- [ ] Pack short sequences: instead of padding to max length, concatenate with `<eos>` markers.

---

## Profiling: how to find the bottleneck

Don't guess. Measure.

- [ ] **`nvidia-smi dmon`**: watch GPU utilization live. Low `sm%` → CPU/data-bound. High `sm%` + low speed → kernel-bound.
- [ ] **`torch.profiler`**: Chrome trace of every op. Best tool for finding unexpected slow ops. [27]
  ```python
  with torch.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
      for _ in range(10): train_step()
  print(p.key_averages().table(sort_by="cuda_time_total"))
  ```
- [ ] **NVIDIA Nsight Systems** (`nsys`): system-level view with CUDA + Python + NCCL. [26]
- [ ] **Time per iteration**: log it. If it jumps, something changed.
- [ ] Wrap ops in `torch.cuda.synchronize()` before timing — async kernels will lie. [27]

---

## Reference: speedups from Karpathy's video (GPT-2 124M on A100) [1]

Baseline → optimized, cumulative:

| Step | Change | tok/s | Speedup |
|------|--------|-------|---------|
| 0 | FP32 baseline | ~18k | 1.0× |
| 1 | `set_float32_matmul_precision('high')` (TF32) | ~45k | 2.5× |
| 2 | + `autocast(bfloat16)` | ~70k | 3.9× |
| 3 | + `torch.compile(model)` | ~130k | 7.2× |
| 4 | + Flash Attention (SDPA) | ~160k | 8.9× |
| 5 | + vocab padding 50257 → 50304 | ~170k | 9.4× |
| 6 | + fused AdamW + grad clip + LR schedule | ~180k | 10.0× |

Numbers independently verified by Towards Data Science step-by-step reproduction on A100. [29]

**Moral:** the first five lines of optimization are essentially free and give a **10× speedup**. Do them before anything fancy. [1][29]

---

## What NOT to do

- [ ] Don't use `DataParallel`. Ever. Use `DistributedDataParallel`. [11][12]
- [ ] Don't call `.item()` or `.cpu()` in the hot loop unless you're logging, and throttle logging. [1]
- [ ] Don't use FP16 on Ampere+ — BF16 is strictly better (same speed, no loss scaling needed). [3][7]
- [ ] Don't quantize optimizer state to INT. Adam's moments need FP precision. [3][17]
- [ ] Don't try to train with INT8/INT4. It's an inference-only trick. [3]
- [ ] Don't write custom attention when SDPA exists. You won't beat Flash Attention. [4][5][8]
- [ ] Don't profile without synchronizing — CUDA is async and timings will be wrong. [27]
- [ ] Don't scale LR linearly forever. Past ~4k batch, switch to sqrt scaling. [20]

---

## References

### Primary source

[1] **Karpathy, A. (2024). "Let's reproduce GPT-2 (124M)."** YouTube video. https://www.youtube.com/watch?v=l8pRSuU81PU
[2] **Karpathy, A. `build-nanogpt` repository.** https://github.com/karpathy/build-nanogpt

### Papers

[3] **Micikevicius, P., et al. (2017). "Mixed Precision Training."** https://arxiv.org/abs/1710.03740
[4] **Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."** https://arxiv.org/abs/2205.14135
[5] **Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."** https://arxiv.org/abs/2307.08691
[12] **Li, S., et al. (2020). "PyTorch Distributed: Experiences on Accelerating Data Parallel Training."** https://arxiv.org/abs/2006.15704
[14] **Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models."** https://arxiv.org/abs/1910.02054
[15] **Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost."** https://arxiv.org/abs/1604.06174
[17] **Dettmers, T., et al. (2021). "8-bit Optimizers via Block-wise Quantization."** https://arxiv.org/abs/2110.02861
[18] **Shazeer, N. & Stern, M. (2018). "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost."** https://arxiv.org/abs/1804.04235
[19] **Chen, X., et al. (2023). "Symbolic Discovery of Optimization Algorithms (Lion)."** https://arxiv.org/abs/2302.06675
[20] **Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour."** https://arxiv.org/abs/1706.02677
[21] **Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3).** https://arxiv.org/abs/2005.14165
[22] **Jiang, A., et al. (2023). "Mistral 7B."** https://arxiv.org/abs/2310.06825
[23] **Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."** https://arxiv.org/abs/2312.00752

### Official docs

[6] **PyTorch — `torch.compile`.** https://pytorch.org/docs/stable/generated/torch.compile.html
[7] **PyTorch — Automatic Mixed Precision (AMP).** https://pytorch.org/docs/stable/amp.html
[8] **PyTorch — `scaled_dot_product_attention`.** https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
[10] **PyTorch — `torch.utils.data.DataLoader`.** https://pytorch.org/docs/stable/data.html
[11] **PyTorch — DistributedDataParallel notes.** https://pytorch.org/docs/stable/notes/ddp.html
[13] **PyTorch — Fully Sharded Data Parallel (FSDP).** https://pytorch.org/docs/stable/fsdp.html
[16] **PyTorch — `torch.utils.checkpoint`.** https://pytorch.org/docs/stable/checkpoint.html
[27] **PyTorch — `torch.profiler`.** https://pytorch.org/docs/stable/profiler.html
[28] **PyTorch — CUDA memory management (`PYTORCH_CUDA_ALLOC_CONF`).** https://pytorch.org/docs/stable/notes/cuda.html#memory-management

### NVIDIA resources

[9] **NVIDIA Deep Learning Performance Guide — Matrix Multiplication Background.** https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
[24] **NVIDIA TransformerEngine (FP8 training).** https://github.com/NVIDIA/TransformerEngine
[25] **NVIDIA NCCL.** https://developer.nvidia.com/nccl
[26] **NVIDIA Nsight Systems.** https://developer.nvidia.com/nsight-systems

### 2024–2025 research (revisions to older best practices)

[29] **Towards Data Science — "Line-By-Line, Let's Reproduce GPT-2, Section 2: Hardware Optimization."** Independent verification of Karpathy's speedup table on A100, with per-step timings. https://towardsdatascience.com/line-by-line-lets-reproduce-gpt-2-section-2-hardware-optimization-86e71c91d9bb/
[30] **"Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention" (2024).** Identifies biased rounding errors in BF16 + Flash Attention that cause systematic gradient corruption when attention probabilities saturate. https://arxiv.org/abs/2510.04212
[31] **Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (2024).** Introduces and validates Warmup-Stable-Decay (WSD) schedule. https://arxiv.org/abs/2405.18392
[32] **Bergsma et al., "Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs" (2025).** Shows linear decay-to-zero outperforms cosine annealing at scale. https://arxiv.org/abs/2502.15938
[33] **Fishman et al., "To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability" (2024).** Demonstrates FP8 convergence gaps and hyperparameter fragility vs BF16. https://arxiv.org/abs/2405.18710
[34] **PyTorch FSDP — activation checkpointing.** Explains why `apply_activation_checkpointing` is required for FSDP (vs raw `torch.utils.checkpoint`). https://pytorch.org/docs/stable/fsdp.html
[35] **Zhao et al., "Deconstructing What Makes a Good Optimizer for Language Models" (2024).** Finds modern optimizers (AdamW, Lion, Adafactor) perform comparably on final loss when tuned; optimizer choice is primarily a memory decision. https://arxiv.org/abs/2407.07972
[36] **Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling" (2024).** Hybrid Mamba + sliding-window attention; trained on 4K, extrapolates to 256K+ with perfect recall. https://arxiv.org/abs/2406.07522

---

## Citation confidence notes

- **[1], [2]** — directly from Karpathy's video + nanoGPT code. Highest confidence.
- **[3]–[5], [12], [14]–[23], [30]–[33], [35], [36]** — published papers, stable arXiv URLs.
- **[6]–[8], [10], [11], [13], [16], [27], [28], [34]** — official PyTorch docs. URLs stable across minor versions; if a page 404s on a future PyTorch release, search `pytorch.org/docs` for the same title.
- **[9], [24]–[26]** — NVIDIA docs / GitHub. Occasionally reorganized; search NVIDIA Developer if a link moves.
- **[29]** — third-party blog post with verified measurements. Treat numbers as approximate.

Before relying on a specific number (e.g., "~2× speedup"), benchmark it on your hardware. Speedups depend on model size, batch size, sequence length, and GPU generation.

## Revision history

- **2026-04-22:** Initial version from Karpathy's GPT-2 video + canonical references.
- **2026-04-22 (rev 2):** Incorporated 2024–2025 findings via web research:
    - Flash Attention + BF16 divergence caveat (ref [30])
    - FP8 training instability (ref [33])
    - WSD / linear-decay LR schedule replaces cosine (refs [31][32])
    - Gradient checkpointing with FSDP requires `apply_activation_checkpointing` (ref [34])
    - Optimizer choice reframed as memory decision, not speed (ref [35])
    - Hybrid Mamba-Attention (Samba) added for long context (ref [36])
    - TF32 speedup tightened from "~3×" to "~2.5×" per measured data (ref [29])
    - `torch.compile` speedup varies by GPU generation
