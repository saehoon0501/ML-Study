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

<!-- Future phases will be appended below -->
