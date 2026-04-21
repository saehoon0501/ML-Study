# AI Engineer Syllabus — Master's Level

### Bridging Software Engineering & Machine Learning

**Profile:** You understand both sides. You can read a paper AND ship a system.
You make better engineering decisions because you understand the math,
and better research decisions because you understand production constraints.

**End goal:** Build your own product — a custom fine-tuned LLM deployed
as an agentic application (LangGraph/LangChain). Every phase feeds this.

**Target roles:** AI Engineer, ML Engineer, Applied ML Scientist
**Total: ~16-20 weeks** at 10-15 hrs/week (master's level depth)

---

## How This Syllabus Works

Each phase has two tracks that reinforce each other:
- **Theory** — understand why things work (and when they break)
- **Systems** — build the thing, deploy the thing, measure the thing

Everything funnels toward one outcome:
**Custom fine-tuned model → agentic app (LangGraph/LangChain) → shipped product**

- P1-3: Understand models deeply enough to choose and customize the right one
- P4: Fine-tune YOUR model for your product's task
- P5: Optimize your agent's retrieval pipeline
- P6: Add multimodal capability (if your product needs it)
- P7: Evaluate your model + agent system rigorously
- P8: Theory behind your production stack
- Portfolio: The product MVP itself

---

## Phase 1: Neural Network Foundations (2 weeks)

The goal isn't to memorize backprop. It's to build intuition for why
training fails, why loss curves look wrong, and what hyperparameters actually do.

### Theory Track

- [ ] 3Blue1Brown — Neural Networks series (Chapters 1-4)
    - https://www.3blue1brown.com/topics/neural-networks
- [ ] Karpathy — "Neural Networks: Zero to Hero" (Videos 1-3)
    - Video 1: micrograd — autograd from scratch
    - Video 2: makemore Part 1 — bigrams, language modeling
    - Video 3: makemore Part 2 — MLP language model
    - https://karpathy.ai/zero-to-hero.html
- [ ] Karpathy — Video 4: Becoming a Backprop Ninja
    - Why: diagnosing training issues requires understanding gradients deeply
- [ ] Karpathy — Video 5: Let's reproduce GPT-2 (**critical**)
    - Real training pipeline, distributed training concepts, end-to-end reproduction
    - Bridges Phase 1 → Phase 2 naturally

### Systems Track

- [ ] PyTorch fundamentals — tensors, autograd, `nn.Module`, dataloaders
    - Official tutorial: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- [ ] **Build:** Train a character-level language model from scratch in PyTorch
    - Not from a tutorial — from your understanding of Videos 1-3

**You're done when:** You can diagnose a NaN loss or a flat training curve
and know where to look (learning rate? initialization? gradient flow?).

---

## Phase 2: Transformers — Architecture & Intuition (2.5 weeks)

This is the core. Every model you'll use in production is a transformer variant.
Understanding the architecture deeply means you can reason about context windows,
attention patterns, why retrieval helps, and what fine-tuning actually changes.

### Theory Track

- [ ] 3Blue1Brown — Transformer videos (Chapters 5-7)
    - Chapter 5: "What is a GPT?" https://www.3blue1brown.com/lessons/gpt
    - Chapter 6: "Attention in Transformers" https://www.3blue1brown.com/lessons/attention
    - Chapter 7: "How LLMs store facts" (MLP layers)
- [ ] Karpathy — Video 6: Building GPT from scratch (**critical**)
- [ ] Karpathy — Video 7: Tokenization (BPE)
    - Why: tokenization choices directly affect model behavior in production
- [ ] Jay Alammar — "The Illustrated Transformer"
    - https://jalammar.github.io/illustrated-transformer/
- [ ] Harvard — "The Annotated Transformer"
    - Line-by-line implementation of "Attention is All You Need"
    - https://nlp.seas.harvard.edu/annotated-transformer/
- [ ] **Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
    - Read alongside the Annotated Transformer
    - https://arxiv.org/abs/1706.03762
- [ ] Modern architecture components (post-2017 additions now standard in all LLMs)
    - RoPE (Rotary Position Embeddings) — replaces sinusoidal PE
    - GQA (Grouped Query Attention) — Llama 2/3/4, Mistral; being challenged by MLA (Multi-Head Latent Attention, DeepSeek) which achieves 93% KV cache reduction
    - FlashAttention (now v4, March 2026) — efficient attention computation
    - RMSNorm (replaces LayerNorm), SwiGLU activation (replaces ReLU/GELU)
    - Resource: Umar Jamil's YouTube deep-dives on each component

### Systems Track

- [ ] Transformer Explainer — interactive tool (Georgia Tech)
    - https://poloclub.github.io/transformer-explainer/
- [ ] **Build:** Implement a small GPT model from scratch in PyTorch
    - Embedding → positional encoding → multi-head attention → FFN → layer norm
    - Train it on a small corpus. Understand every line.
- [ ] DeepLearning.AI — "Attention in Transformers" short course
    - https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch

**You're done when:** You can whiteboard Q/K/V attention, explain why
`sqrt(d_k)` scaling matters, why RoPE replaced sinusoidal PE, how GQA
reduces KV cache, and what Flash Attention changes — AND you've built one that trains.

---

## Phase 3: How LLMs Actually Work at Scale (2 weeks)

Bridge phase. You understand the architecture — now understand how it
scales, how it's trained on internet-scale data, and why that matters
for the models you'll use in production.

### Theory Track

- [ ] **Paper (skim):** GPT-3 "Language Models are Few-Shot Learners" (Brown et al., 2020)
    - Focus on: scaling behavior, in-context learning, few-shot prompting
    - Skip: most of the benchmark tables
- [ ] **CS336 selected lectures** (not the full course):
    - Lecture 1: Tokenization (reinforces Karpathy Video 7)
    - Lecture 3: Architectures & Hyperparameters
    - Lecture 9: Scaling Laws — understand Chinchilla, compute-optimal training
    - Lecture 10: Inference — KV cache, speculative decoding, batching
    - Lecture 12: Evaluation
    - https://stanford-cs336.github.io/spring2025/
- [ ] **Paper (skim):** Chinchilla "Training Compute-Optimal LLMs" (Hoffmann et al., 2022)
    - Core insight: most models are over-parameterized and under-trained
    - **Post-Chinchilla update:** Llama 3 showed "inference-optimal" training (train smaller models longer) supersedes Chinchilla for deployment. Read the Llama 3.1 technical report for this.
- [ ] Mixture of Experts (MoE) — now dominant in frontier models
    - How sparse routing works, why MoE enables massive models with modest compute
    - DeepSeek-V3/R1 technical reports: Multi-head Latent Attention (MLA), cost-efficient MoE, reasoning models
    - Mixtral as the accessible open-source example
- [ ] State-space models awareness (Mamba/Mamba-2, Jamba hybrid)
    - Not dominant yet, but architecturally important for interviews

### Systems Track

- [ ] Understand inference optimization: KV cache, continuous batching, PagedAttention
    - Read vLLM blog post on PagedAttention
    - SGLang as emerging alternative (RadixAttention for shared-prefix workloads)
- [ ] **Exercise:** Given a model size and GPU specs, calculate:
    - Memory required for inference (weights + KV cache)
    - Expected throughput (tokens/sec)
    - Whether it fits on your hardware

**You're done when:** Someone announces a new 70B model and you can
estimate its memory footprint, predict roughly how it'll perform vs. a 7B model,
and explain what "compute-optimal" means.

---

## Phase 4: Fine-Tuning — Theory & Practice (2.5 weeks)

This is where ML knowledge meets engineering. You're not just calling
`SFTTrainer` — you understand what LoRA is doing mathematically,
why rank-4 vs rank-16 matters, and when fine-tuning is the wrong approach.

### Theory Track

- [ ] **Paper:** LoRA (Hu et al., 2021)
    - Core insight: weight updates during fine-tuning are low-rank
    - Understand: why low-rank decomposition works, rank selection tradeoffs
- [ ] **Paper:** DoRA (Liu et al., 2024) — Weight-Decomposed Low-Rank Adaptation
    - Production-ready in PEFT, consistently outperforms LoRA at same rank
    - https://arxiv.org/abs/2402.09353
- [ ] **Paper:** QLoRA (Dettmers et al., 2023)
    - 4-bit quantization + LoRA = fine-tune large models on small GPUs
    - Understand: NF4 quantization, double quantization, paged optimizers
- [ ] Alignment methods: understand the landscape
    - SFT (supervised fine-tuning) — when and why
    - DPO (Direct Preference Optimization) — the practical default for alignment
    - ORPO, KTO — simpler alternatives gaining traction (TRL supports both natively)
    - RLHF — conceptual understanding only (rarely used directly in practice now)
- [ ] **Decision framework:** When NOT to fine-tune
    - Prompting > few-shot > RAG > fine-tuning (escalation ladder)
    - Read: Hamel Husain's blog posts on this decision

### Systems Track

- [ ] HF LLM Course — Chapters 1-7 (Transformers, Tokenizers, Datasets, pipeline)
    - https://huggingface.co/learn/llm-course/en/chapter1/1
- [ ] Key libraries deep dive:
    - `transformers` — model loading, inference, tokenization
    - `datasets` — loading and processing training data
    - `peft` — LoRA, DoRA, QLoRA configuration and tradeoffs
    - `trl` — SFTTrainer, DPO/ORPO/KTO trainers
    - `accelerate` — multi-GPU training
    - **Unsloth** — 2-5x faster fine-tuning, 60-80% less VRAM. Use as default for QLoRA workflows.
    - **Argilla** — data annotation and curation (the hardest part of fine-tuning in practice)
- [ ] HF LLM Course Chapter 11 — Supervised Fine-Tuning
    - https://huggingface.co/learn/llm-course/chapter11/3
- [ ] Phil Schmid — "How to Fine-Tune Open LLMs in 2025"
    - https://www.philschmid.de/fine-tune-llms-in-2025
- [ ] **Project: Fine-tune a model with QLoRA + Unsloth for a specific task**
    - Pick a model: **Qwen 3 8B** (Apache 2.0, best benchmarks) or **Llama 4 Scout** (17B MoE, largest ecosystem)
    - Prepare a custom dataset with Argilla (not a canned one — practice data curation)
    - Quality over quantity: 1K high-quality examples often beats 50K noisy ones
    - Train with DoRA, different ranks, compare results
    - Get chat templates right (the #1 silent failure mode in fine-tuning)
    - Experiment with model merging (mergekit — SLERP/TIES/DARE)
    - Push to HF Hub

| Method | VRAM | Example GPU | Trainable Params |
| --- | --- | --- | --- |
| Full fine-tune (7B) | ~120 GB | 2x H100 | 100% |
| LoRA (7B) | ~24 GB | RTX 4090 | ~1% |
| QLoRA (7B) | ~16 GB | RTX 3090/4080 | ~1% (4-bit base) |

**You're done when:** You can decide whether a problem needs fine-tuning
vs. RAG vs. prompting, choose the right approach, execute it,
and explain the tradeoffs to a non-ML engineer.

---

## Phase 5: RAG & Retrieval — Theory Gaps Only (1 week)

You build RAG systems daily at LG CNS with LangChain/LangGraph.
This phase fills the theoretical gaps behind what you already do in production.

### Theory Track (focus area)

- [ ] Embedding models — how text becomes vectors
    - Foundations: Sentence-BERT, E5, BGE (baselines to understand)
    - Current SOTA: Gemini Embedding 001, Qwen3-Embedding-8B, Jina v5, Voyage AI — check MTEB leaderboard
    - Note: MTEB v2 (2026) scores ≠ MTEB v1 — not directly comparable
    - https://huggingface.co/spaces/mteb/leaderboard
    - Understand: contrastive learning (InfoNCE loss), cosine similarity, embedding dimensions
    - Why some queries "miss" relevant documents (semantic gap)
- [ ] Vector search internals
    - ANN algorithms: HNSW, IVF, Product Quantization — how they actually work
    - Tradeoff: recall vs. latency vs. memory
- [ ] Reranking deep dive
    - Cross-encoders vs. bi-encoders: why cross-encoders are better but slower
    - ColBERT late interaction pattern
- [ ] **Paper (skim):** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- [ ] Advanced RAG patterns you may not have used
    - HyDE (Hypothetical Document Embeddings), Self-RAG, Corrective RAG
    - GraphRAG (Microsoft, 2024) — graph-based community summarization for global queries
    - RAPTOR (Stanford, 2024) — recursive abstractive tree-structured retrieval
    - Late Chunking (Jina AI) — embed long docs before chunking to preserve context
    - Agentic RAG — agents that plan retrieval steps dynamically (theory, not implementation)
- [ ] RAG evaluation beyond RAGAS
    - ARES (automated RAG eval, Stanford)
    - Text-to-SQL as a retrieval pattern (structured + semantic hybrid)
- [ ] Context engineering for agents
    - Read: Anthropic "Effective context engineering for AI agents"

### Skipped (you already know this)

- ~~LangChain / LangGraph fundamentals~~ — daily work
- ~~Vector database hands-on~~ — daily work
- ~~Build a RAG system project~~ — you've built many
- ~~DeepLearning.AI LangChain course~~ — below your level

**You're done when:** You can explain the THEORY behind the RAG systems you
already build — why HNSW works, why cross-encoders beat bi-encoders,
what contrastive learning does. Not just "it works" but "here's why."

---

## Phase 6: Multimodal AI (1 week, conditional)

**Keep if** your product involves images, audio, or mixed media.
**Skip if** your product is text + structured data only — redirect
those 7 days to the portfolio artifact build.

### Theory Track

- [ ] Vision Transformers (ViT) — how transformers process images
    - Patch embedding, position encoding for 2D
- [ ] Vision-Language Models (VLMs)
    - Historical: CLIP, LLaVA, Flamingo (understand the architecture patterns)
    - Current SOTA: **GLM-4.5V** (106B MoE, top benchmarks), **Qwen2.5-VL-72B**, Molmo
    - How vision encoders connect to language models
    - Contrastive vs. generative multimodal training
- [ ] **ColPali** — vision-based document retrieval (bypasses OCR + chunking entirely)
    - Bridges Phase 5 (RAG) and Phase 6 — highly relevant to your daily RAG work
    - https://arxiv.org/abs/2407.01449
- [ ] Document understanding
    - OCR-free models: GOT-OCR2.0, DocOwl 2
    - Audio: Whisper v3, Qwen2-Audio (awareness level)

### Systems Track

- [ ] HF `transformers` multimodal pipelines
    - Image captioning, visual QA, image-text matching
- [ ] CLIP embeddings for image-text retrieval
- [ ] **Project: Multimodal product understanding system**
    - Input: product image + text description + structured attributes
    - Tasks: categorization, quality assessment, duplicate detection
    - Use a VLM (**Qwen2.5-VL** or similar) + structured data integration

**You're done when:** You can design a system that takes in a product image,
its description, and its specs, and makes intelligent decisions about
categorization, quality, or matching.

---

## Phase 7: Evaluation & Experimentation (2 weeks)

The JD emphasizes "정량 평가" and "오프라인/온라인 실험."
This is where many AI engineers are weakest — building is easy,
knowing if it actually works is hard.

### Theory Track

- [ ] LLM benchmarks — what they measure and their limitations
    - MMLU, HumanEval, MATH, BBH, MT-Bench, Chatbot Arena
    - Read: "Top LLM Benchmarks Explained" (Confident AI)
    - https://www.confident-ai.com/blog/llm-benchmarks-mmlu-hellaswag-and-beyond
- [ ] Evaluation methodology for applied AI
    - Offline evaluation: held-out test sets, cross-validation
    - LLM-as-judge: using stronger models to evaluate weaker ones
    - Human evaluation design: inter-annotator agreement, rubrics
- [ ] **Agent evaluation** (critical — most AI engineers skip this)
    - Paper: "Survey on Evaluation of LLM-based Agents" (2025) — covers planning, tool use, memory
    - CLEAR Framework: Cost, Latency, Efficacy, Assurance, Reliability for enterprise agents
    - Read: Hamel Husain "Selecting the Right AI Evals Tool" — hands-on comparison of tools
    - Read: Hamel Husain "LLM Evals FAQ" — best practitioner guide
- [ ] Red-teaming & safety evaluation
    - Anthropic Bloom — automated behavioral eval generation (open-source)
    - Promptfoo — open-source automated red-teaming with adversarial prompts
    - Paper: "Fine-Tuning Lowers Safety" — critical for your fine-tuning work
- [ ] Experimentation & causal inference basics
    - A/B testing: statistical significance, sample size, guardrail metrics
    - Offline policy evaluation (for ranking/recommendation)
    - When online experiments aren't feasible: backtesting approaches
- [ ] Continuous evaluation in production
    - Input drift detection (PSI, KL divergence, embedding drift)
    - Output quality monitoring over time
    - Resource: Evidently AI LLM evaluation framework (open-source)
- [ ] Explore HF Open LLM Leaderboard
    - https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

### Systems Track

- [ ] Evaluation tools:
    - `lm-evaluation-harness` (EleutherAI) — standardized benchmarks
    - `deepeval` — custom evaluation pipelines
    - `ragas` — RAG-specific evaluation (faithfulness, relevance, recall)
    - **Braintrust** — best for CI/CD eval gating + production observability
    - **Arize Phoenix** — best open-source observability option
    - https://deepeval.com/docs/benchmarks-introduction
- [ ] **Project: Evaluation pipeline for your agentic system**
    - Build a test dataset with ground-truth answers
    - Measure: retrieval recall, answer correctness, faithfulness, latency
    - Compare: different chunking strategies, embedding models, retrieval methods
    - Present results as a decision framework, not just numbers

| Benchmark | What It Tests | When You Care |
| --- | --- | --- |
| MMLU / MMLU-Pro | General knowledge (57 subjects) | Comparing base models |
| HumanEval / MBPP | Code generation | Code-related tasks |
| MATH Level 5 | Mathematical reasoning | Math-heavy applications |
| MT-Bench / Arena | Conversation quality | Chat/assistant use cases |
| Custom domain eval | Your specific task | Always — this matters most |

**You're done when:** You can design an evaluation framework for a new
AI feature, run offline experiments, interpret results with statistical
rigor, and make a go/no-go recommendation with evidence.

---

## Phase 8: Production Theory (1 week)

You deploy AI systems daily at LG CNS. This phase covers only the
theory you haven't formalized — no hands-on deployment projects.

### Theory Track (focus area)

- [ ] Quantization theory
    - GPTQ, AWQ, GGUF — how each works mathematically
    - FP8 (Hopper/Blackwell GPUs), AWQ-Marlin as 2026 production standard
    - GGUF still relevant for CPU/hybrid inference (llama.cpp, Ollama)
    - Precision vs. quality tradeoffs, when to use which
- [ ] Inference optimization theory
    - KV caching internals, continuous batching, speculative decoding, PagedAttention
    - Framework landscape: **vLLM** (default), **SGLang** (shared-prefix), TensorRT-LLM (raw throughput). TGI is in maintenance mode.
    - Cost modeling: $/1M tokens, multi-tier model routing, semantic caching
- [ ] LLM observability (pick one, know the landscape)
    - **Langfuse** (open-source, self-hostable), LangSmith (best with LangChain/LangGraph)
    - Helicone (fastest setup, cost tracking), Arize Phoenix (drift detection)
- [ ] MLOps theory
    - Model versioning, output quality drift detection
    - CI/CD for ML: testing model outputs, regression detection
    - Read: huyenchip.com/mlops
- [ ] MCP (Model Context Protocol) — awareness level
    - Industry-standard protocol for connecting LLMs to external tools/data (97M monthly SDK downloads)
    - Governed by Linux Foundation (AAIF), adopted by Anthropic, OpenAI, Microsoft, AWS, Google
    - Understand: MCP servers (external tool integration) vs Skills (markdown instructions) — complementary, not competing
    - Relevant to your agent tool design and LangGraph production patterns
- [ ] Human-in-the-loop design patterns
    - Confidence routing, feedback loops, active learning
    - LangGraph 1.0+ (stable Oct 2025): interrupt/resume for human-in-the-loop, PostgresSaver
    - Read: Anthropic "Effective harnesses for long-running agents"

### Skipped (you already do this)

- ~~Inference frameworks hands-on~~ — you deploy with these
- ~~Deploy an end-to-end service project~~ — this is your day job
- ~~MLOps hands-on (MLflow, Docker)~~ — you know this

**You're done when:** You can explain the theory behind your production
systems — why PagedAttention helps, how drift detection works,
when to route to humans. Interview-ready depth.

---

## Portfolio Artifact: Your Product MVP (5-6 weeks)

**This serves two purposes simultaneously:**
1. The #1 deliverable for your master's applications
2. The first version of the product you want to build

### The stack

```
Your custom fine-tuned model (Phase 4)
        ↓
Agentic framework (LangGraph/LangChain) — your existing expertise
        ↓
RAG pipeline with optimized retrieval (Phase 5 theory)
        ↓
Evaluation framework (Phase 7)
        ↓
Deployed product with API + demo
```

### Product domain

TBD — to be decided during Phase 7 (Day 92). By then you'll have:
- Fine-tuned a model (Phase 4)
- Deepened RAG theory (Phase 5)
- Built an eval framework (Phase 7)
- Enough signal to pick a domain that fits your skills + market gap

### What makes it portfolio-worthy AND product-worthy

- Custom model, not just API wrapper (fine-tuned for specific task)
- Agentic architecture with real tool use and multi-step reasoning
- Rigorous evaluation proving it works (not just vibes)
- Clean, well-documented code (admissions will read it; users will fork it)
- Working demo (Gradio/Streamlit)
- Solves a real problem (not a tutorial project)
- SOP narrative: "I built X at LG CNS, discovered gap Y,
  created Z to address it — now I want to go deeper at [school]"

### Structure

- **Phase A (2 weeks):** Architecture, core model + agent pipeline, tests, v0.1
- **Phase B (2.5 weeks):** Eval results, advanced features, demo, documentation, v1.0
- **Phase C (Sep-Oct):** User feedback, iteration, growth toward real product

---

## Enrichment Materials (use to deepen each phase)

Materials marked with a phase number indicate where they fit best.
Use these when you want to go deeper on a topic, not as required reading.

### DeepLearning.AI Short Courses (Free)

All available at https://www.deeplearning.ai/courses/

| Course | Phase | What it adds |
| --- | --- | --- |
| Attention in Transformers (w/ StatQuest) | P2 | Already in syllabus — code self-attention in PyTorch |
| LangChain for LLM Application Development | P5 | Already in syllabus — chains, agents, tool use |
| Building and Evaluating Advanced RAG | P5 | Already in syllabus — advanced retrieval patterns |
| **Building Multimodal Search and RAG** | P5-P6 | Multimodal retrieval — bridges RAG and multimodal phases |
| **LangChain: Chat with Your Data** | P5 | RAG over private documents, deeper than the LangChain intro |
| **Multi AI Agent Systems with crewAI** | P5 | Multi-agent orchestration — useful for complex commerce workflows |
| **Building and Evaluating Data Agents** | P5-P7 | Multi-agent + data sources + evaluation |
| **Agent Memory: Building Memory-Aware Agents** | P5 | Long-term memory for agents — session persistence |
| **Agentic AI** (Andrew Ng) | P5 | Agentic design patterns: reflection, tool use, planning |
| **Fine-tuning and Reinforcement Learning for LLMs** | P4 | Reinforcement learning for shaping model behavior |
| **Semantic Caching for AI Agents** | P8 | Production optimization — reduce latency and cost |

### UC Berkeley — LLM Agents (Free MOOC)

- **LLM Agents Course (Fall 2024)**: https://llmagents-learning.org/f24
    - Guest lectures from Google DeepMind, OpenAI, Anthropic, NVIDIA, Meta
    - Covers: chain-of-thought, ReAct, compound AI systems, safety
    - **Best for:** Phase 5 (agents) and Phase 8 (production systems)
- **Advanced LLM Agents (Spring 2025)**: https://agenticai-learning.org/sp25
    - Advanced reasoning, code generation, program verification
    - **Best for:** After Phase 5, if going deep on agents
- **Agentic AI (Fall 2025)**: https://rdi.berkeley.edu/agentic-ai/f25
    - Agentic frameworks, infrastructure, planning
    - **Best for:** Phase 8 (production agent systems)

### Anthropic Engineering Blog (Free, Essential Reading)

https://www.anthropic.com/engineering — Read these as you reach the relevant phase.

| Post | Phase | Why it matters |
| --- | --- | --- |
| **Building effective agents** (Dec 2024) | P5 | Start simple, add complexity only when needed |
| **Effective context engineering for AI agents** (Sep 2025) | P5 | Context > prompting — the real skill |
| **Writing effective tools for agents** (Sep 2025) | P5-P8 | How to design tools that agents can actually use well |
| **Introducing Contextual Retrieval** (Sep 2024) | P5 | Anthropic's approach to improving RAG retrieval |
| **The "think" tool** (Mar 2025) | P5 | Letting agents pause and reason in complex tool use |
| **Demystifying evals for AI agents** (Jan 2026) | P7 | Practical evaluation strategies for production agents |
| **Designing AI-resistant technical evaluations** (Jan 2026) | P7 | How to build evals that actually test what you think |
| **Effective harnesses for long-running agents** (Nov 2025) | P8 | Production patterns for agent reliability |
| **Advanced tool use on Claude Developer Platform** (Nov 2025) | P8 | Production tool use patterns |

### Prompt Engineering (Free)

- **Prompt Engineering Guide**: https://www.promptingguide.ai/
    - 18+ techniques: Chain-of-Thought, ReAct, Tree-of-Thought, structured outputs
    - **Best for:** Phase 5 (before building agents)
- **Anthropic's Prompt Engineering docs**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
    - Anthropic-specific best practices
    - **Best for:** Phase 5-8

### Hands-On Cookbooks (Free)

- **Claude Cookbooks**: https://github.com/anthropics/claude-cookbooks
    - Jupyter notebooks: customer support agents, RAG systems, vision apps
    - **Best for:** Phase 5-6 projects
- **OpenAI Cookbook**: https://cookbook.openai.com/
    - Production RAG, multi-agent orchestration, function calling
    - **Best for:** Phase 5 projects
- **LlamaIndex Resources**: https://developers.llamaindex.ai/
    - 70+ recipes: ReAct agents, text-to-SQL, semantic search
    - **Best for:** Phase 5 (alternative to LangChain perspective)

### MLOps & Production (Free)

- **MLOps Zoomcamp** (DataTalks.Club): https://github.com/DataTalksClub/mlops-zoomcamp
    - Free 3-month course: MLflow, Docker, monitoring with Evidently AI
    - YouTube playlist + assignments + certification
    - **Best for:** Phase 8 deep dive
- **Chip Huyen — "Designing Machine Learning Systems"** (O'Reilly, book)
    - The definitive book on production ML systems design
    - **Best for:** Read during Phase 8, reference throughout
- **Stanford CS 329S — ML Systems Design**: https://stanford-cs329s.github.io/
    - Chip Huyen's Stanford course — lectures available
    - **Best for:** Phase 8
- **Evidently AI — ML Observability Course** (Free, 7 weeks)
    - Focus on monitoring ML models in production
    - https://www.evidentlyai.com/blog/mlops-courses
    - **Best for:** Phase 8 monitoring deep dive
- **Chip Huyen's MLOps Guide**: https://huyenchip.com/mlops/
    - Free web resource covering the full MLOps landscape

### Evaluation Deep Dives

- **Sebastian Raschka — "4 Approaches to LLM Evaluation"**
    - https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches
    - **Best for:** Phase 7
- **Braintrust** — offline eval + production observability platform
    - `autoevals` library: pre-built scorers for factuality, relevance
    - **Best for:** Phase 7 tooling
- **LMSYS Chatbot Arena**: https://chat.lmsys.org/
    - Most trusted human-preference benchmark (Arena ELO)
    - **Best for:** Phase 7 (understanding real-world model comparison)

### Vector Databases Deep Dive

- **Pinecone Learning Center**: https://www.pinecone.io/learn/
    - FAISS tutorial, vector search fundamentals, ANN algorithms
    - **Best for:** Phase 5 (Day 67-73)
- **Activeloop GenAI Courses**: https://learn.activeloop.ai/
    - 68+ lessons on LangChain, vector databases, advanced RAG
    - **Best for:** Phase 5 (alternative structured learning path)

### Multimodal Deep Dive

- **VLM Guide (Rohit Bandaru)**: https://rohitbandaru.github.io/blog/Vision-Language-Models/
    - Comprehensive overview of VLM architectures and evolution
    - **Best for:** Phase 6 theory
- **BentoML Multimodal AI Guide**: https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models
    - Practical guide to open-source VLMs in 2026
    - **Best for:** Phase 6 model selection

### Stanford Courses (reference)

- **CS224N — NLP with Deep Learning**: https://web.stanford.edu/class/cs224n/
    - Excellent if you want deeper NLP foundations (embeddings → transformers)
- **CS25 — Transformers United**: https://web.stanford.edu/class/cs25/
    - Seminar format, great for staying current on research directions
- **CS336 — remaining lectures** (if curious about training at scale)
    - https://stanford-cs336.github.io/spring2025/

### Books

- **Dive into Deep Learning (d2l.ai)**: http://www.d2l.ai/
    - Free, runnable code, good reference for specific topics

### Reading Lists

- Sebastian Raschka's LLM reading list: https://sebastianraschka.com/blog/2023/llm-reading-list.html

### Fine-Tuning Advanced

- **Unsloth**: https://github.com/unslothai/unsloth — fastest QLoRA workflows
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl — YAML-config reproducibility
- **mergekit**: https://github.com/arcee-ai/mergekit — model merging (SLERP/TIES/DARE)
- **Argilla**: https://argilla.io — data curation and annotation
- Sebastian Raschka "Build a Large Language Model (From Scratch)" (Manning, 2024)
- Sebastian Raschka "Build a Reasoning Model (From Scratch)" (Manning, pub Jul 2026) — sequel covering reasoning/RL
- Sebastian Raschka's fine-tuning articles: https://magazine.sebastianraschka.com
- Sebastian Raschka MLA supplement chapter: https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/

### Key References Added by Research (2026)

- Lilian Weng's blog: https://lilianweng.github.io — "The Transformer Family v2", attention mechanisms
- CMU 11-711 / 11-667 lecture slides — shows what CMU expects students to know
- Hamel Husain eval guides: https://hamel.dev/blog/posts/eval-tools/ and https://hamel.dev/blog/posts/evals-faq/
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard — current embedding model rankings

---

## Progress Tracker

| Phase | Topic | Weeks | Status |
| --- | --- | --- | --- |
| 1 | Neural Network Foundations | 2.3 | In progress |
| 2 | Transformers Architecture | 2.7 | Not started |
| 3 | LLMs at Scale | 1.7 | Not started |
| 4 | Fine-Tuning | 2.4 | Not started |
| 5 | RAG Theory Gaps | 1 | Not started |
| 6 | Multimodal AI | 1 | Not started |
| 7 | Evaluation & Experimentation | 2.3 | Not started |
| 8 | Production Theory | 0.7 | Not started |
| P | Portfolio Artifact | 4.6 | Not started |
| | **Total** | **~19** | |

---

## How This Maps to Master's Applications

| Application Component | Covered By |
| --- | --- |
| Interview: "Explain attention/transformers" | Phase 2 (whiteboard-ready) |
| Interview: "Explain scaling laws" | Phase 3 (Chinchilla, compute-optimal) |
| Interview: "Walk me through fine-tuning" | Phase 4 (LoRA/QLoRA theory + hands-on) |
| Interview: "How do you evaluate AI systems?" | Phase 7 (deepest phase) |
| Resume: Open-source project | Portfolio artifact |
| SOP: Technical depth + impact | Artifact + LG CNS work + eval expertise |
| Differentiator vs other applicants | Production agent exp + eval framework |

---

*Last updated: 2026-04-09 (enriched with 6-agent parallel research review)*
