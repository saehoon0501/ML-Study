# Daily Study Timeline — 2hrs/day (Revised for Master's Application)

**Start:** 2026-04-09 (Thu) | **End:** ~2026-08-31 (Mon)
**Pace:** 2 hours every day, 7 days/week
**Total:** ~145 days (~21 weeks)

Each day flows into the next. Watch sessions include note-taking.
Code sessions mean hands-on implementation, not just reading.

### End goal

**Custom fine-tuned model → agentic app (LangGraph/LangChain) → shipped product**

Every phase feeds this. The portfolio artifact IS the product MVP.

### What changed from v1

- **P1-P4 unchanged** — foundational depth to choose and customize the right model
- **P5 (RAG) compressed 18 → 7 days** — theory gaps only, you build RAG daily at LG CNS
- **P6 (Multimodal) compressed 10 → 7 days** — conditional, skip if product doesn't need it
- **P7 (Evaluation) expanded 13 → 16 days** — evaluate YOUR model + agent system
- **P8 (Production) compressed 14 → 5 days** — theory only, you deploy daily
- **Portfolio Artifact (46 days)** — your product MVP, also serves as admissions portfolio

### v3 enrichment (6-agent research + web-verified corrections, Apr 2026)

- **P1-P2:** Added Karpathy Video 5, modern arch components (RoPE, GQA, Flash Attention, RMSNorm, SwiGLU)
- **P3:** Added MoE (DeepSeek-V3/R1), post-Chinchilla (Llama 3 inference-optimal), SGLang
- **P4:** Unsloth default, DoRA > LoRA (r=16 + use_dora=True), ORPO/KTO, Argilla, mergekit. Models: **Qwen 3 8B** or **Llama 4 Scout**
- **P5:** Updated embeddings (Gemini Embed, Qwen3-Embedding, Jina v5), GraphRAG/RAPTOR/Late Chunking/ColPali, ARES
- **P6:** GLM-4.5V + Qwen2.5-VL-72B as current VLM leaders, document understanding (OCR-free), audio awareness
- **P7:** Agent eval papers, Bloom (confirmed Dec 2025), Promptfoo, Hamel Husain guides, Braintrust ($80M raise), DeepEval
- **P8:** FP8/AWQ-Marlin, SGLang 29% faster than vLLM on H100, MCP (97M downloads, Linux Foundation), LangGraph 1.0+, observability

### Hard deadlines driving this timeline

```
Jun-Jul 2026 .... TOEFL retake (2 attempts)
Aug-Sep 2026 .... GRE prep + take (CMU MSAII only)
Oct 2026 ........ Portfolio artifact shipped
Nov 2026 ........ SOP drafting
Dec 10, 2026 .... CMU MSAII deadline
Dec 15, 2026 .... USC MSCS deadline
Jan 2027 ........ Cornell Tech, Duke AIPI, Northwestern deadlines
```

---

## Phase 1: Neural Network Foundations (unchanged)

### Week 1 (Apr 9 - Apr 15)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 1 | Apr 9 (Thu) | 3B1B Neural Networks | Watch Ch 1-2: What is a NN + Gradient Descent. Take notes on weight/bias intuition |
| 2 | Apr 10 (Fri) | 3B1B Neural Networks | Watch Ch 3-4: Backpropagation + Backprop calculus. Map chain rule to computation graph |
| 3 | Apr 11 (Sat) | Karpathy micrograd (1/2) | Watch Video 1 first half (~1.5hrs). Follow along: build Value class, implement forward pass |
| 4 | Apr 12 (Sun) | Karpathy micrograd (2/2) | Watch Video 1 second half. Implement backward pass, build simple neural net on top of autograd |
| 5 | Apr 13 (Mon) | micrograd — own implementation | Close the tutorial. Rebuild micrograd from scratch from memory. Debug until it works |
| 6 | Apr 14 (Tue) | Karpathy makemore p1 (1/2) | Watch Video 2 first half. Bigram model, counting, probability tables |
| 7 | Apr 15 (Wed) | Karpathy makemore p1 (2/2) | Watch Video 2 second half. Loss function (negative log likelihood), intro to neural net approach |

### Week 2 (Apr 16 - Apr 22)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 8 | Apr 16 (Thu) | Karpathy makemore p2 (1/2) | Watch Video 3 first half. MLP architecture, embedding layer, hidden layer |
| 9 | Apr 17 (Fri) | Karpathy makemore p2 (2/2) | Watch Video 3 second half. Training loop, learning rate tuning, train/val split |
| 10 | Apr 18 (Sat) | Karpathy backprop ninja (1/2) | Watch Video 4 first half. Manual backprop through each operation |
| 11 | Apr 19 (Sun) | Karpathy backprop ninja (2/2) | Watch Video 4 second half. Complete manual gradient derivations. Understand why this matters for debugging |
| 12 | Apr 20 (Mon) | PyTorch fundamentals (1/2) | PyTorch 60-min blitz: tensors, autograd. Compare with your micrograd |
| 13 | Apr 21 (Tue) | PyTorch fundamentals (2/2) | `nn.Module`, dataloaders, training loop patterns. Rewrite makemore MLP in idiomatic PyTorch |
| 14 | Apr 22 (Wed) | Karpathy GPT-2 reproduction | Watch Video 5 (condensed): real training pipeline, distributed training. **Critical** bridge P1→P2 |

### Week 3 (Apr 23 - Apr 24) — Phase 1 finish

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 15 | Apr 23 (Thu) | **Build:** char-level LM (1/2) | Start from blank file. Design architecture, implement model class, training loop, loss tracking |
| 16 | Apr 24 (Fri) | **Build:** char-level LM (2/2) | Debug, evaluate, generate samples. Experiment: different hidden sizes, learning rates |

**Checkpoint:** You can train a neural net from scratch, understand every gradient, and diagnose training issues.

---

## Phase 2: Transformers — Architecture & Intuition (unchanged)

### Week 3 cont'd (Apr 25 - Apr 29)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 17 | Apr 25 (Sat) | 3B1B Transformers | Watch Ch 5: "What is a GPT?" — high-level transformer architecture |
| 18 | Apr 26 (Sun) | 3B1B Transformers | Watch Ch 6: "Attention in Transformers" — Q/K/V mechanics, attention patterns |
| 19 | Apr 27 (Mon) | 3B1B Transformers + Illustrated Transformer | Watch Ch 7: "How LLMs store facts" (MLP layers). Start reading Jay Alammar's Illustrated Transformer |
| 20 | Apr 28 (Tue) | Illustrated Transformer | Finish Illustrated Transformer. Draw the full architecture diagram yourself |
| 21 | Apr 29 (Wed) | Karpathy GPT from scratch (1/4) | Watch Video 6 first quarter (~30min). Setup, data loading, bigram baseline |

### Week 4 (Apr 30 - May 6)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 22 | Apr 30 (Thu) | Karpathy GPT (2/4) | Watch Video 6: self-attention mechanism. Code single-head attention, then multi-head |
| 23 | May 1 (Fri) | Karpathy GPT (3/4) | Watch Video 6: feed-forward layers, residual connections, layer norm |
| 24 | May 2 (Sat) | Karpathy GPT (4/4) | Watch Video 6: complete transformer block, training. Run the full model |
| 25 | May 3 (Sun) | Karpathy GPT — replay & code | Re-implement the GPT from Video 6 without watching. Fill in gaps from memory |
| 26 | May 4 (Mon) | Karpathy Tokenization | Watch Video 7: BPE tokenization. Understand why tokenization affects model behavior (e.g., numbers, multilingual) |
| 27 | May 5 (Tue) | Transformer Explainer + DL.AI (1/2) | Play with Georgia Tech Transformer Explainer (30min). Start DeepLearning.AI "Attention in Transformers" course |
| 28 | May 6 (Wed) | DL.AI Attention course (2/2) | Finish attention course. Code self-attention from scratch in PyTorch (not copying) |

### Week 5 (May 7 - May 13)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 29 | May 7 (Thu) | "Attention Is All You Need" paper | Read Sections 1-7. Map each concept to what you already built. Start Annotated Transformer |
| 30 | May 8 (Fri) | Annotated Transformer | Finish Harvard's Annotated Transformer. Compare their implementation with yours from Video 6 |
| 31 | May 9 (Sat) | Modern architecture components | RoPE, GQA/MLA, FlashAttention v4, RMSNorm, SwiGLU — how modern LLMs differ from the 2017 paper. Watch Umar Jamil's videos on these |
| 32 | May 10 (Sun) | **Build:** GPT from scratch (1/4) | Start YOUR implementation. Not Karpathy's, not Harvard's — yours. Design the architecture |
| 33 | May 11 (Mon) | **Build:** GPT (2/4) | Implement multi-head attention + positional encoding. Unit test attention outputs |
| 34 | May 12 (Tue) | **Build:** GPT (3/4) | Implement transformer blocks, full model. Write training loop |
| 35 | May 13 (Wed) | **Build:** GPT (4/4) | Train on a small corpus (Shakespeare, code, etc). Generate samples. Debug until satisfied |

**Checkpoint:** You can whiteboard the full transformer architecture and have your own working implementation.

---

## Phase 3: How LLMs Actually Work at Scale (unchanged)

### Week 6 (May 14 - May 20)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 36 | May 14 (Thu) | GPT-3 paper (1/2) | Read Sections 1-3: scaling, architecture decisions, few-shot prompting framework |
| 37 | May 15 (Fri) | GPT-3 paper (2/2) | Skim Sections 4-6: focus on in-context learning results, skip benchmark tables. Note what changes from GPT to GPT-3 |
| 38 | May 16 (Sat) | CS336 Lecture 1 | Tokenization: BPE variants, SentencePiece, how tokenizer choice affects downstream performance |
| 39 | May 17 (Sun) | CS336 Lecture 3 | Architectures & Hyperparameters: RoPE, GQA, SwiGLU, RMSNorm — modern transformer choices |
| 40 | May 18 (Mon) | CS336 Lecture 9 (1/2) | Scaling Laws part 1: Kaplan et al., power laws, compute-optimal training |
| 41 | May 19 (Tue) | CS336 L9 (2/2) + Chinchilla + Llama 3 | Finish scaling laws. Chinchilla paper (skim) + Llama 3.1 report: post-Chinchilla "inference-optimal" training |
| 42 | May 20 (Wed) | MoE + DeepSeek | Mixture of Experts: sparse routing, why MoE enables massive models cheaply. DeepSeek-V3/R1 reports: MLA, cost-efficient MoE, reasoning models |

### Week 7 (May 21 - May 25)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 43 | May 21 (Thu) | CS336 Lecture 10 + 12 | Inference (KV cache, batching, speculative decoding) + Evaluation (benchmarks, contamination) |
| 44 | May 22 (Fri) | Inference optimization deep dive | Read vLLM PagedAttention blog. SGLang RadixAttention. Understand KV cache memory layout, why paging helps |
| 45 | May 23 (Sat) | **Exercise:** memory calculations (1/2) | Calculate: 7B model memory footprint (FP16, INT8, INT4). KV cache size for different sequence lengths |
| 46 | May 24 (Sun) | **Exercise:** memory calculations (2/2) | Calculate: throughput estimates for different batch sizes. Given a GPU (A100 80GB), what models fit? What's the max context? |
| 47 | May 25 (Mon) | Review & consolidation | Review all Phase 3 notes. Create a personal "model evaluation cheat sheet" — what to check when a new model drops |

**Checkpoint:** You can estimate memory/throughput for any model, explain scaling laws, and evaluate new model releases critically.

---

## Phase 4: Fine-Tuning — Theory & Practice (unchanged)

### Week 7 cont'd + Week 8 (May 26 - Jun 3)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 48 | May 26 (Tue) | LoRA paper | Read LoRA paper. Understand low-rank decomposition, rank selection, which layers to apply to |
| 49 | May 27 (Wed) | DoRA + QLoRA papers | DoRA (weight-decomposed LoRA, 2024) — outperforms LoRA. QLoRA: NF4, double quantization, paged optimizers |
| 50 | May 28 (Thu) | When to fine-tune | Decision framework: prompting > few-shot > RAG > fine-tuning. Read Hamel Husain's posts on this |
| 51 | May 29 (Fri) | Alignment: SFT + DPO + ORPO | SFT fundamentals. DPO (practical default). ORPO/KTO (simpler alternatives). RLHF (conceptual only) |
| 52 | May 30 (Sat) | Data curation for fine-tuning | Quality > quantity. Argilla for annotation. Deduplication (MinHash). Chat template formatting — the #1 silent failure |
| 53 | May 31 (Sun) | HF LLM Course Ch 1-2 | Course overview, Transformer models: using pre-trained models, understanding the pipeline API |
| 54 | Jun 1 (Mon) | HF LLM Course Ch 3-4 | Tokenizers deep dive (connects to Karpathy Video 7). Fine-tuning fundamentals |
| 55 | Jun 2 (Tue) | HF LLM Course Ch 5-6 | Datasets library: loading, processing, streaming. Working with large datasets |
| 56 | Jun 3 (Wed) | HF LLM Course Ch 7 | Main NLP tasks: classification, NER, QA. Using `pipeline()` for inference |

### Week 9 (Jun 4 - Jun 11)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 57 | Jun 4 (Thu) | HF LLM Course Ch 11 | Supervised Fine-Tuning walkthrough. Understand SFTTrainer, data formatting, chat templates |
| 58 | Jun 5 (Fri) | Phil Schmid guide | "How to Fine-Tune Open LLMs in 2025" — end-to-end practical guide with modern HF stack |
| 59 | Jun 6 (Sat) | Libraries deep dive | Hands-on with `peft` (DoRA/LoRA config), `trl` (SFT/DPO trainers), **Unsloth** (default for QLoRA). Run examples |
| 60 | Jun 7 (Sun) | **Project:** dataset curation | Pick your task. Curate with Argilla (not a canned dataset). Clean, format, split. Quality over quantity |
| 61 | Jun 8 (Mon) | **Project:** fine-tune attempt 1 | Fine-tune **Qwen 3 8B** or **Llama 4 Scout** with Unsloth + QLoRA + DoRA, rank=16. Track loss, eval metrics |
| 62 | Jun 9 (Tue) | **Project:** fine-tune attempt 2 | Change rank (16, 64). Try LoRA vs DoRA. Compare results. Try different learning rates |
| 63 | Jun 10 (Wed) | **Project:** model merging + push | Best runs → experiment with mergekit (SLERP/TIES). Push best model to HF Hub |
| 64 | Jun 11 (Thu) | **Project:** wrap-up | Test inference. Write up findings. Document decision framework: when fine-tuning vs RAG vs prompting |

**Checkpoint:** You know when to fine-tune vs. prompt vs. RAG, and you've shipped a fine-tuned model to HF Hub.

---

## Phase 5: RAG & Retrieval — Theory Gaps Only (compressed: 18 → 7 days)

You build RAG systems daily at LG CNS. This phase covers only what you likely haven't formalized: embedding theory, ANN algorithm internals, reranking math, and advanced patterns.

### Week 10 (Jun 12 - Jun 18)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 65 | Jun 12 (Fri) | Embedding models theory | Foundations (S-BERT, E5, BGE) → current SOTA (Gemini Embed, Qwen3-Embedding, Jina v5). Check MTEB v2 leaderboard. Contrastive learning (InfoNCE). Read: Anthropic "Contextual Retrieval" |
| 66 | Jun 13 (Sat) | Vector search internals | HNSW, IVF, Product Quantization — how they actually work. Recall vs latency vs memory tradeoffs. Reference: Pinecone FAISS tutorial |
| 67 | Jun 14 (Sun) | Reranking deep dive | Cross-encoders vs bi-encoders, ColBERT late interaction. **ColPali**: vision-based doc retrieval bypassing OCR — bridges P5 and P6 |
| 68 | Jun 15 (Mon) | Advanced RAG patterns | HyDE, Self-RAG, Corrective RAG, **GraphRAG** (Microsoft), **RAPTOR** (Stanford), **Late Chunking** (Jina). Agentic RAG theory |
| 69 | Jun 16 (Tue) | RAG paper + evaluation | RAG paper (Lewis et al., 2020). RAGAS + **ARES** (automated RAG eval). Text-to-SQL as structured retrieval pattern |
| 70 | Jun 17 (Wed) | Context engineering | Read: Anthropic "Effective context engineering for AI agents." Map to your LG CNS agent work — what would you change? |
| 71 | Jun 18 (Thu) | RAG theory review | Synthesize: create a decision tree for "given X retrieval problem, use Y approach." Compare with your production experience at LG CNS |

**Checkpoint:** You understand the theory behind the RAG systems you already build. You can explain WHY your production choices work, not just THAT they work.

---

## Phase 6: Multimodal AI (compressed: 10 → 7 days)

### Week 11 (Jun 19 - Jun 25)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 72 | Jun 19 (Fri) | Vision Transformers (ViT) | How transformers process images: patch embedding, 2D position encoding. Read ViT paper intro |
| 73 | Jun 20 (Sat) | CLIP + VLMs | Contrastive pretraining. Historical: LLaVA, Flamingo. Current: **GLM-4.5V** (SOTA MoE), **Qwen2.5-VL-72B**, Molmo. How vision encoders connect to LLMs |
| 74 | Jun 21 (Sun) | Document understanding | OCR-free models (GOT-OCR2.0, DocOwl 2). ColPali for vision-based retrieval. Audio awareness: Whisper v3, Qwen2-Audio |
| 75 | Jun 22 (Mon) | HF multimodal pipelines | Hands-on: image captioning, visual QA, image-text matching, CLIP embeddings for retrieval |
| 76 | Jun 23 (Tue) | **Project:** multimodal system (1/3) | Design: product image + description + attributes → categorization/matching system |
| 77 | Jun 24 (Wed) | **Project:** multimodal (2/3) | Implement pipeline. Connect VLM (GLM-4.5V or Qwen2.5-VL) + structured data integration |
| 78 | Jun 25 (Thu) | **Project:** multimodal (3/3) | Evaluate accuracy, handle edge cases, document. Push to GitHub |

**Checkpoint:** You can build systems that reason over images, text, and structured data together.

---

## Phase 7: Evaluation & Experimentation (expanded: 13 → 16 days)

This is your highest-leverage phase for the master's application. Evaluation is:
- The skill most AI engineers lack
- The foundation for your portfolio artifact
- What CMU MSAII and Cornell Tech will probe in interviews

### Week 12 (Jun 26 - Jul 2)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 79 | Jun 26 (Fri) | LLM benchmarks overview | Read "Top LLM Benchmarks Explained" (Confident AI). Sebastian Raschka "4 Approaches to LLM Evaluation." MMLU, HumanEval, MATH, BBH — what each tests and its limitations |
| 80 | Jun 27 (Sat) | Benchmarks deep dive | Explore HF Open LLM Leaderboard. LMSYS Chatbot Arena. Understand why Arena ELO is most trusted. What makes benchmarks saturate? |
| 81 | Jun 28 (Sun) | Agent evaluation | Read: Anthropic "Demystifying evals for AI agents" + "Survey on Evaluation of LLM-based Agents" (2025). CLEAR framework (Cost/Latency/Efficacy/Assurance/Reliability) |
| 82 | Jun 29 (Mon) | LLM-as-judge + red-teaming | LLM-as-judge: calibration, bias. Red-teaming: **Anthropic Bloom** (automated behavioral evals) + **Promptfoo** (adversarial prompts). Read: "Designing AI-resistant technical evaluations" |
| 83 | Jun 30 (Tue) | Human eval + practitioner guides | Rubrics, inter-annotator agreement. Read: Hamel Husain "Selecting the Right AI Evals Tool" + "LLM Evals FAQ" |
| 84 | Jul 1 (Wed) | Eval tools: lm-eval-harness + Braintrust | EleutherAI's harness (benchmarks). **Braintrust** (CI/CD eval gating). **Arize Phoenix** (open-source observability) |
| 85 | Jul 2 (Thu) | Eval tools: deepeval + ragas | Set up `deepeval` for custom evals. `ragas` + ARES for RAG evaluation. Build a simple eval pipeline |

### Week 13 (Jul 3 - Jul 9)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 86 | Jul 3 (Fri) | A/B testing + continuous eval | Statistical significance, sample size, guardrail metrics. Production monitoring: drift detection (PSI, embedding drift), Evidently AI |
| 87 | Jul 4 (Sat) | Experimentation advanced | Offline policy evaluation. Backtesting when online experiments aren't feasible. Cost-aware evaluation (CostBench) |
| 88 | Jul 5 (Sun) | **Project:** eval framework (1/4) | Design evaluation framework for an agentic AI system (based on your LG CNS work). Define task taxonomy, metrics, test cases |
| 89 | Jul 6 (Mon) | **Project:** eval framework (2/4) | Implement automated evaluation: task completion rate, tool use accuracy, reasoning chain quality |
| 90 | Jul 7 (Tue) | **Project:** eval framework (3/4) | Add LLM-as-judge scoring. Compare with human annotations. Measure correlation |
| 91 | Jul 8 (Wed) | **Project:** eval framework (4/4) | Run comparative experiments: different agent architectures, different prompting strategies. Produce a decision report |
| 92 | Jul 9 (Thu) | Portfolio artifact scoping | Based on what you built in the eval project: define the open-source artifact scope. What's the reusable framework? What's the README? |

### Week 14 (Jul 10 - Jul 11)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 93 | Jul 10 (Fri) | Eval playbook | Create your personal evaluation playbook: given any new AI feature, here's how to evaluate it. This document is interview gold |
| 94 | Jul 11 (Sat) | Phase 7 review | Review all eval material. Prepare to explain: benchmark limitations, LLM-as-judge calibration, agent eval challenges |

**Checkpoint:** You can design rigorous evaluations, run experiments, and make data-driven go/no-go decisions. Your eval framework is the seed for the portfolio artifact.

---

## Phase 8: Production Theory (compressed: 14 → 5 days)

You deploy AI systems daily at LG CNS. This phase covers only theory you haven't formalized.

### Week 14 cont'd (Jul 12 - Jul 16)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 95 | Jul 12 (Sun) | Quantization theory | GPTQ, AWQ, GGUF, **FP8** (Hopper/Blackwell), AWQ-Marlin. GGUF for CPU/hybrid (llama.cpp/Ollama). Precision vs quality tradeoffs |
| 96 | Jul 13 (Mon) | Inference + observability | Frameworks: **vLLM** (default), **SGLang** (shared-prefix), TensorRT-LLM (raw throughput). TGI → maintenance. Observability: Langfuse, LangSmith, Arize Phoenix |
| 97 | Jul 14 (Tue) | MLOps + MCP | Model versioning, drift detection, CI/CD for ML. **MCP** (Model Context Protocol) — new interop standard for LLM tooling. Read: huyenchip.com/mlops |
| 98 | Jul 15 (Wed) | Human-in-the-loop + LangGraph 1.0 | Confidence routing, feedback loops. LangGraph 1.0+: interrupt/resume for HITL, PostgresSaver. Read: Anthropic "Effective harnesses for long-running agents" |
| 99 | Jul 16 (Thu) | Production review | Synthesize: system design template for "given X AI feature, here's the production architecture." Cost optimization: multi-tier model routing, semantic caching |

**Checkpoint:** You can explain the theory behind the production systems you already build.

---

## Portfolio Artifact: Your Product MVP (46 days)

**This serves two purposes:**
1. The #1 deliverable for your master's applications
2. The first version of the product you want to build

### The stack

```
Custom fine-tuned model (from Phase 4)
    ↓
Agentic framework (LangGraph/LangChain — your existing expertise)
    ↓
RAG pipeline with optimized retrieval (Phase 5 theory applied)
    ↓
Evaluation proving it works (Phase 7 framework)
    ↓
Deployed product with API + demo
```

### Product domain

TBD — decide on Day 92 after completing the eval framework project.
By then you'll have fine-tuned a model, deepened RAG theory, and built
an eval framework. Enough signal to pick a domain.

### Phase A: Architecture & Core (Jul 17 - Jul 31, 15 days)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 100 | Jul 17 (Fri) | Architecture design | Define product scope, model requirements, agent architecture, API surface. Draw the system diagram |
| 101 | Jul 18 (Sat) | Project setup | GitHub repo, CI/CD, linting, testing framework, documentation structure |
| 102 | Jul 19 (Sun) | Model integration | Load your fine-tuned model (Phase 4). Build inference pipeline. Test baseline quality |
| 103 | Jul 20 (Mon) | Agent pipeline | Build LangGraph/LangChain agent around your model. Tool definitions, state management |
| 104 | Jul 21 (Tue) | RAG integration | Wire retrieval pipeline into agent. Apply Phase 5 theory: reranking, hybrid search |
| 105 | Jul 22 (Wed) | Unit tests | Comprehensive test coverage for model, agent, and retrieval modules |
| 106 | Jul 23 (Thu) | Evaluation integration | Wire Phase 7 eval framework into the product. Automated quality metrics |
| 107 | Jul 24 (Fri) | Core review & refactor | Review all core code. Refactor for clarity. This is portfolio code — it must be clean |
| 108 | Jul 25 (Sat) | Integration tests | End-to-end: input → agent → model → retrieval → output → eval |
| 109 | Jul 26 (Sun) | API layer | FastAPI (your expertise) — user-facing API for the product |
| 110 | Jul 27 (Mon) | Documentation (1/2) | README: installation, quickstart, architecture overview |
| 111 | Jul 28 (Tue) | Documentation (2/2) | API reference, examples, contributing guide |
| 112 | Jul 29 (Wed) | Bug fixes + edge cases | Run against diverse inputs. Fix failures. Harden |
| 113 | Jul 30 (Thu) | v0.1 release | Tag release, write changelog. Artifact is usable by others |
| 114 | Jul 31 (Fri) | Dogfood | Use your own tool on a real problem. Note friction points |

### Phase B: Polish & Demonstrate (Aug 1 - Aug 17, 17 days)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 115 | Aug 1 (Sat) | Iteration based on dogfooding | Fix UX issues found during dogfooding |
| 116 | Aug 2 (Sun) | Advanced feature 1 | Add differentiated feature (e.g., cost tracking, LLM-as-judge integration) |
| 117 | Aug 3 (Mon) | Advanced feature 2 | Add second differentiator (e.g., visualization dashboard, comparative reports) |
| 118 | Aug 4 (Tue) | Advanced feature 3 | Add third differentiator or refine existing ones |
| 119 | Aug 5 (Wed) | Tests for new features | Full test coverage for advanced features |
| 120 | Aug 6 (Thu) | Performance optimization | Profile, benchmark, optimize hot paths |
| 121 | Aug 7 (Fri) | Real-world case study (1/3) | Apply artifact to a real scenario. Document setup, process, results |
| 122 | Aug 8 (Sat) | Real-world case study (2/3) | Run full evaluation. Produce quantitative results |
| 123 | Aug 9 (Sun) | Real-world case study (3/3) | Write up findings. This becomes the "Results" section of your README |
| 124 | Aug 10 (Mon) | README rewrite | Polish README with real results, architecture diagrams, comparison tables |
| 125 | Aug 11 (Tue) | Demo: Gradio/Streamlit | Build interactive demo for the artifact |
| 126 | Aug 12 (Wed) | v0.2 release | Tag release with all improvements |
| 127 | Aug 13 (Thu) | Community prep | Prepare for sharing: write HN/Reddit post draft, identify relevant Discord/Slack communities |
| 128 | Aug 14 (Fri) | Code review pass | Final code quality review. Every function documented. No dead code |
| 129 | Aug 15 (Sat) | Resume integration | Update resume with the artifact. Quantify: lines of code, test coverage, features, case study results |
| 130 | Aug 16 (Sun) | SOP connection | Draft the paragraph in your SOP that references this artifact. How does it demonstrate your capability? |
| 131 | Aug 17 (Mon) | v1.0 release | Final release. Artifact is portfolio-ready |

### Phase C: Product Growth (Sep-Oct, outside daily timeline)

After Aug 17, the product continues independently (no daily schedule needed):

- [ ] Get real users — share on relevant communities, get feedback
- [ ] Iterate based on user feedback (not hypothetical features)
- [ ] Track usage metrics for resume quantification
- [ ] Explore monetization or market fit signals
- [ ] **October 2026:** Product is mature enough to reference in all applications
- [ ] Continue developing toward a real product beyond admissions

---

## Calendar Summary

| Month | Dates | Phases | Days |
|-------|-------|--------|------|
| Apr | 9-30 | Phase 1 + Phase 2 start | 22 |
| May | 1-31 | Phase 2 finish + Phase 3 + Phase 4 start | 31 |
| Jun | 1-30 | Phase 4 finish + Phase 5 + Phase 6 + Phase 7 start | 30 |
| Jul | 1-31 | Phase 7 finish + Phase 8 + Portfolio start | 31 |
| Aug | 1-17 | Portfolio artifact polish + v1.0 | 17 |

## Phase Boundaries

| Phase | Start | End | Days | Change |
|-------|-------|-----|------|--------|
| 1. NN Foundations | Apr 9 | Apr 24 | 16 | — |
| 2. Transformers | Apr 25 | May 13 | 19 | — |
| 3. LLMs at Scale | May 14 | May 25 | 12 | — |
| 4. Fine-Tuning | May 26 | Jun 11 | 17 | — |
| 5. RAG Theory Gaps | Jun 12 | Jun 18 | 7 | **-11 days** |
| 6. Multimodal | Jun 19 | Jun 25 | 7 | **-3 days** |
| 7. Evaluation | Jun 26 | Jul 11 | 16 | **+3 days** |
| 8. Production Theory | Jul 12 | Jul 16 | 5 | **-9 days** |
| Portfolio Artifact | Jul 17 | Aug 17 | 32 | **NEW (replaces 12-day capstone)** |
| | | **Total** | **131** | |

---

## How This Maps to Master's Applications

| Application Component | Covered By |
|----------------------|------------|
| **Interview: "Explain attention"** | Phase 2 (whiteboard-ready) |
| **Interview: "Explain scaling laws"** | Phase 3 (Chinchilla, compute-optimal) |
| **Interview: "Walk me through fine-tuning"** | Phase 4 (LoRA/QLoRA theory + hands-on) |
| **Interview: "How do you evaluate AI systems?"** | Phase 7 (deepest phase, interview gold) |
| **Resume: Open-source project** | Portfolio artifact (stars, usage, case study) |
| **SOP: Technical depth + impact** | Artifact + LG CNS work + Phase 7 eval expertise |
| **Differentiator vs other applicants** | Production agent experience (LG CNS) + evaluation framework (rare) |

---

## Parallel Tracks (not in 2hr/day budget)

These happen alongside the study schedule on separate time:

| Track | When | Notes |
|-------|------|-------|
| TOEFL prep + retake | Jun-Jul 2026 | 2 attempts booked |
| GRE prep (CMU only) | Aug-Sep 2026 | After portfolio Phase A |
| WES ICAP | Start Jul 2026 | 6-8 week processing |
| Recommender outreach | Aug 2026 | 3 LORs needed |
| SOP drafting | Nov 2026 | After artifact v1.0 |

---

## Rules for Yourself

1. **2 hours means 2 hours.** Timer on. No "just 10 more minutes" — consistency beats intensity.
2. **If you finish early,** review yesterday's notes. Don't jump ahead.
3. **If you fall behind,** skip the review/consolidation days, not the build days.
4. **Every project must produce a GitHub commit.** Not "I understood it" — code or it didn't happen.
5. **Weekend = normal days.** The schedule already accounts for 7 days/week.
6. **If stuck for >30min,** move on and revisit tomorrow with fresh eyes.
7. **The portfolio artifact is the #1 priority after Phase 4.** If something must be cut, cut study days, not build days.
8. **TOEFL and GRE are separate time blocks.** Don't let test prep eat into the 2hr study budget.

---

*Revised: 2026-04-21 — Updated for master's application timeline*
