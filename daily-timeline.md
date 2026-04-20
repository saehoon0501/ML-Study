# Daily Study Timeline — 2hrs/day

**Start:** 2026-04-09 (Thu) | **End:** ~2026-08-17 (Mon)
**Pace:** 2 hours every day, 7 days/week
**Total:** ~131 days (~19 weeks)

Each day flows into the next. Watch sessions include note-taking.
Code sessions mean hands-on implementation, not just reading.

---

## Phase 1: Neural Network Foundations

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
| 14 | Apr 22 (Wed) | **Build:** char-level LM (1/3) | Start from blank file. Design architecture based on what you learned. Implement model class |

### Week 3 (Apr 23 - Apr 26) — Phase 1 finish

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 15 | Apr 23 (Thu) | **Build:** char-level LM (2/3) | Implement training loop, loss tracking, learning rate schedule. Start training |
| 16 | Apr 24 (Fri) | **Build:** char-level LM (3/3) | Debug, evaluate, generate samples. Experiment: what happens with different hidden sizes, learning rates? |

**Checkpoint:** You can train a neural net from scratch, understand every gradient, and diagnose training issues.

---

## Phase 2: Transformers — Architecture & Intuition

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
| 29 | May 7 (Thu) | "Attention Is All You Need" paper (1/2) | Read Sections 1-3 of the paper. Map each concept to what you already built |
| 30 | May 8 (Fri) | Paper + Annotated Transformer (1/2) | Read Sections 4-7 of paper. Start Harvard's Annotated Transformer side by side |
| 31 | May 9 (Sat) | Annotated Transformer (2/2) | Finish Annotated Transformer. Compare their implementation with yours from Video 6 |
| 32 | May 10 (Sun) | **Build:** GPT from scratch (1/4) | Start YOUR implementation. Not Karpathy's, not Harvard's — yours. Design the architecture |
| 33 | May 11 (Mon) | **Build:** GPT (2/4) | Implement multi-head attention + positional encoding. Unit test attention outputs |
| 34 | May 12 (Tue) | **Build:** GPT (3/4) | Implement transformer blocks, full model. Write training loop |
| 35 | May 13 (Wed) | **Build:** GPT (4/4) | Train on a small corpus (Shakespeare, code, etc). Generate samples. Debug until satisfied |

**Checkpoint:** You can whiteboard the full transformer architecture and have your own working implementation.

---

## Phase 3: How LLMs Actually Work at Scale

### Week 6 (May 14 - May 20)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 36 | May 14 (Thu) | GPT-3 paper (1/2) | Read Sections 1-3: scaling, architecture decisions, few-shot prompting framework |
| 37 | May 15 (Fri) | GPT-3 paper (2/2) | Skim Sections 4-6: focus on in-context learning results, skip benchmark tables. Note what changes from GPT to GPT-3 |
| 38 | May 16 (Sat) | CS336 Lecture 1 | Tokenization: BPE variants, SentencePiece, how tokenizer choice affects downstream performance |
| 39 | May 17 (Sun) | CS336 Lecture 3 | Architectures & Hyperparameters: RoPE, GQA, SwiGLU, RMSNorm — modern transformer choices |
| 40 | May 18 (Mon) | CS336 Lecture 9 (1/2) | Scaling Laws part 1: Kaplan et al., power laws, compute-optimal training |
| 41 | May 19 (Tue) | CS336 Lecture 9 (2/2) + Chinchilla | Finish scaling laws. Read Chinchilla paper (skim): key insight about data vs parameters |
| 42 | May 20 (Wed) | CS336 Lecture 10 | Inference: KV cache mechanics, continuous batching, speculative decoding |

### Week 7 (May 21 - May 27)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 43 | May 21 (Thu) | CS336 Lecture 12 | Evaluation: benchmarks, contamination, how to measure model quality |
| 44 | May 22 (Fri) | Inference optimization deep dive | Read vLLM PagedAttention blog post. Understand KV cache memory layout, why paging helps |
| 45 | May 23 (Sat) | **Exercise:** memory calculations (1/2) | Calculate: 7B model memory footprint (FP16, INT8, INT4). KV cache size for different sequence lengths |
| 46 | May 24 (Sun) | **Exercise:** memory calculations (2/2) | Calculate: throughput estimates for different batch sizes. Given a GPU (A100 80GB), what models fit? What's the max context? |
| 47 | May 25 (Mon) | Review & consolidation | Review all Phase 3 notes. Create a personal "model evaluation cheat sheet" — what to check when a new model drops |

> **Enrichment for Phase 3:** If you want more depth on inference systems, read the vLLM paper (Kwon et al., 2023) and Anthropic's blog on "Effective harnesses for long-running agents."

**Checkpoint:** You can estimate memory/throughput for any model, explain scaling laws, and evaluate new model releases critically.

---

## Phase 4: Fine-Tuning — Theory & Practice

### Week 7 cont'd + Week 8 (May 26 - Jun 3)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 48 | May 26 (Tue) | LoRA paper (1/2) | Read LoRA paper Sections 1-4. Understand low-rank decomposition, why weight updates are low-rank |
| 49 | May 27 (Wed) | LoRA paper (2/2) | Finish LoRA paper. Understand rank selection, which layers to apply LoRA to, ablation results |
| 50 | May 28 (Thu) | QLoRA paper | Read QLoRA paper. Focus on: NF4 data type, double quantization, paged optimizers. Understand the engineering tricks |
| 51 | May 29 (Fri) | Alignment landscape | Read about SFT, DPO, RLHF conceptually. CS336 alignment lecture or Chip Huyen's RLHF blog |
| 52 | May 30 (Sat) | Alignment continued | DPO vs RLHF tradeoffs. When to use which. Read the DPO paper intro (Rafailov et al., 2023) |
| 53 | May 31 (Sun) | HF LLM Course Ch 1-2 | Course overview, Transformer models: using pre-trained models, understanding the pipeline API |
| 54 | Jun 1 (Mon) | HF LLM Course Ch 3-4 | Tokenizers deep dive (connects to Karpathy Video 7). Fine-tuning fundamentals |
| 55 | Jun 2 (Tue) | HF LLM Course Ch 5-6 | Datasets library: loading, processing, streaming. Working with large datasets |
| 56 | Jun 3 (Wed) | HF LLM Course Ch 7 | Main NLP tasks: classification, NER, QA. Using `pipeline()` for inference |

### Week 9 (Jun 4 - Jun 10)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 57 | Jun 4 (Thu) | HF LLM Course Ch 11 | Supervised Fine-Tuning walkthrough. Understand SFTTrainer, data formatting, chat templates |
| 58 | Jun 5 (Fri) | Phil Schmid guide | "How to Fine-Tune Open LLMs in 2025" — end-to-end practical guide with modern HF stack |
| 59 | Jun 6 (Sat) | Libraries deep dive | Hands-on with `peft` (LoRA config), `trl` (SFTTrainer), `accelerate`. Run example scripts |
| 60 | Jun 7 (Sun) | **Project:** dataset curation | Pick your task. Curate a custom dataset (not a canned one). Clean, format, split train/val/test |
| 61 | Jun 8 (Mon) | **Project:** fine-tune attempt 1 | Fine-tune a 7B model (Llama 3 / Mistral / Qwen) with QLoRA, rank=4. Track loss, eval metrics |
| 62 | Jun 9 (Tue) | **Project:** fine-tune attempt 2 | Change LoRA rank (16, 64). Compare results. Try different learning rates |
| 63 | Jun 10 (Wed) | **Project:** analysis & push | Compare all runs. Analyze: what rank works best? Why? Push best model to HF Hub |
| 64 | Jun 11 (Thu) | **Project:** wrap-up | Test inference on your fine-tuned model. Write up findings. Document the decision framework |

> **Enrichment for Phase 4:** DL.AI "Fine-tuning and Reinforcement Learning for LLMs" short course. Also read Sebastian Raschka's LoRA variants guide if you want to explore DoRA, rsLoRA.

**Checkpoint:** You know when to fine-tune vs. prompt vs. RAG, and you've shipped a fine-tuned model to HF Hub.

---

## Phase 5: RAG & Retrieval Systems

### Week 10 (Jun 12 - Jun 18)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 65 | Jun 12 (Fri) | Embedding models theory | How text → vectors works. Sentence-BERT, E5, BGE architectures. Contrastive learning intuition. Read: Anthropic "Introducing Contextual Retrieval" blog |
| 66 | Jun 13 (Sat) | Embeddings hands-on | Generate embeddings with `sentence-transformers`. Visualize with t-SNE/UMAP. Test cosine similarity on real queries |
| 67 | Jun 14 (Sun) | Vector search fundamentals | HNSW, IVF algorithms. Understand recall vs latency vs memory tradeoffs. Reference: Pinecone Learning Center FAISS tutorial |
| 68 | Jun 15 (Mon) | Retrieval strategies | Dense vs sparse (BM25) vs hybrid. Cross-encoder reranking vs bi-encoder. When each wins |
| 69 | Jun 16 (Tue) | Chunking strategies | Fixed-size, semantic, recursive chunking. How chunk size affects retrieval quality. Experiment with different strategies |
| 70 | Jun 17 (Wed) | RAG paper + advanced patterns (1/2) | Skim original RAG paper (Lewis et al., 2020). Then: HyDE, query decomposition |
| 71 | Jun 18 (Thu) | Advanced RAG (2/2) | Self-RAG, Corrective RAG, multi-step retrieval. When simple RAG isn't enough |

### Week 11 (Jun 19 - Jun 25)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 72 | Jun 19 (Fri) | Vector DBs: FAISS | Hands-on with FAISS. Build an index, add vectors, search. Understand index types (Flat, IVF, HNSW) |
| 73 | Jun 20 (Sat) | Vector DBs: production-grade | Hands-on with Chroma or Qdrant. Compare API, persistence, metadata filtering with FAISS |
| 74 | Jun 21 (Sun) | LangChain fundamentals (1/3) | Chains, prompts, output parsers. Build a simple chain. Pre-read: Prompt Engineering Guide (promptingguide.ai) CoT & ReAct sections |
| 75 | Jun 22 (Mon) | LangChain (2/3) | Agents, tool use, structured outputs. Read: Anthropic "Building effective agents" blog + "Writing effective tools for agents" |
| 76 | Jun 23 (Tue) | LangGraph | Multi-step workflows, state machines. Build a simple agent graph. Read: Anthropic "The think tool" blog |
| 77 | Jun 24 (Wed) | DL.AI: LangChain course | "LangChain for LLM Application Development" — work through the full short course |
| 78 | Jun 25 (Thu) | DL.AI: Advanced RAG | "Building and Evaluating Advanced RAG" — work through the full short course |

### Week 12 (Jun 26 - Jul 2)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 79 | Jun 26 (Fri) | **Project:** RAG system (1/4) | Design architecture. Set up ingestion pipeline: document loading → chunking → embedding |
| 80 | Jun 27 (Sat) | **Project:** RAG (2/4) | Build retrieval layer: vector store + BM25 hybrid search + metadata filtering |
| 81 | Jun 28 (Sun) | **Project:** RAG (3/4) | Add reranking (cross-encoder). Wire up generation with retrieved context. Test end-to-end |
| 82 | Jun 29 (Mon) | **Project:** RAG (4/4) | Evaluate: retrieval recall, answer faithfulness, latency. Fix the weakest link. Document |

> **Enrichment for Phase 5:** (1) DL.AI "Building Multimodal Search and RAG" — bridges into Phase 6. (2) DL.AI "Multi AI Agent Systems with crewAI" — multi-agent patterns. (3) DL.AI "Agentic AI" by Andrew Ng. (4) UC Berkeley LLM Agents MOOC (llmagents-learning.org/f24) — watch selected lectures on ReAct and compound AI systems. (5) Claude Cookbooks (github.com/anthropics/claude-cookbooks) — try the RAG and agent notebooks. (6) Anthropic "Effective context engineering for AI agents" blog. (7) LlamaIndex resources (developers.llamaindex.ai) for alternative RAG perspectives.

**Checkpoint:** You can build, diagnose, and fix a production-grade RAG system. You know why retrieval fails and how to fix it.

---

## Phase 6: Multimodal AI

### Week 12 cont'd + Week 13 (Jun 30 - Jul 9)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 83 | Jun 30 (Tue) | Vision Transformers (ViT) | How transformers process images: patch embedding, 2D position encoding. Read ViT paper intro. Reference: rohitbandaru.github.io/blog/Vision-Language-Models |
| 84 | Jul 1 (Wed) | CLIP | Contrastive language-image pretraining. How CLIP connects vision and text. Play with OpenCLIP |
| 85 | Jul 2 (Thu) | Vision-Language Models | LLaVA, Flamingo architecture patterns. How vision encoders connect to LLMs |
| 86 | Jul 3 (Fri) | VLMs continued + Structured data | Contrastive vs generative multimodal training. Table understanding, schema-aware prompting, Text-to-SQL basics |
| 87 | Jul 4 (Sat) | HF multimodal pipelines | Hands-on: image captioning, visual QA, image-text matching using `transformers` |
| 88 | Jul 5 (Sun) | CLIP embeddings hands-on | Build an image-text retrieval system using CLIP embeddings. Search product images by text query |
| 89 | Jul 6 (Mon) | **Project:** multimodal system (1/4) | Design: product image + description + attributes → categorization system |
| 90 | Jul 7 (Tue) | **Project:** multimodal (2/4) | Implement image processing pipeline. Connect VLM (LLaVA or similar) for image understanding |
| 91 | Jul 8 (Wed) | **Project:** multimodal (3/4) | Integrate structured data. Build the full classification/matching pipeline |
| 92 | Jul 9 (Thu) | **Project:** multimodal (4/4) | Test on real product data. Evaluate accuracy. Handle edge cases. Document findings |

> **Enrichment for Phase 6:** (1) DL.AI "Multi-Vector Image Retrieval" — fine-grained image-text matching. (2) BentoML Multimodal AI Guide — practical model selection for 2026. (3) DL.AI "Building Multimodal Search and RAG" if not done in Phase 5.

**Checkpoint:** You can build systems that reason over images, text, and structured data together.

---

## Phase 7: Evaluation & Experimentation

### Week 14 (Jul 10 - Jul 16)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 93 | Jul 10 (Fri) | LLM benchmarks overview | Read "Top LLM Benchmarks Explained" (Confident AI). Read Sebastian Raschka "4 Approaches to LLM Evaluation." MMLU, HumanEval, MATH, BBH — what each tests |
| 94 | Jul 11 (Sat) | Benchmarks continued | Explore HF Open LLM Leaderboard. Understand Leaderboard v2 metrics. What makes a good benchmark? Limitations? |
| 95 | Jul 12 (Sun) | Evaluation methodology (1/2) | Offline eval: held-out test sets, cross-validation for LLMs. LLM-as-judge: using strong models to eval weak ones. Read: Anthropic "Demystifying evals for AI agents" blog |
| 96 | Jul 13 (Mon) | Evaluation methodology (2/2) | Human evaluation design: rubrics, inter-annotator agreement. When human eval is necessary vs automated |
| 97 | Jul 14 (Tue) | A/B testing fundamentals | Statistical significance, sample size calculation, guardrail metrics. How to design an online experiment |
| 98 | Jul 15 (Wed) | Experimentation advanced | Offline policy evaluation for ranking/reco. Backtesting when online experiments aren't feasible |
| 99 | Jul 16 (Thu) | Eval tools: lm-evaluation-harness | Hands-on: install EleutherAI's harness. Run a benchmark on a small model. Understand the output |

### Week 15 (Jul 17 - Jul 23)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 100 | Jul 17 (Fri) | Eval tools: deepeval + ragas | Set up `deepeval` for custom evals. Set up `ragas` for RAG evaluation (faithfulness, relevance, recall) |
| 101 | Jul 18 (Sat) | **Project:** eval pipeline (1/4) | Design evaluation framework for your Phase 5 RAG system. Define metrics, build test dataset with ground truth |
| 102 | Jul 19 (Sun) | **Project:** eval pipeline (2/4) | Implement automated evaluation: retrieval recall, answer correctness, faithfulness scores |
| 103 | Jul 20 (Mon) | **Project:** eval pipeline (3/4) | Run comparative experiments: different chunking strategies, different embedding models, different retrieval methods |
| 104 | Jul 21 (Tue) | **Project:** eval pipeline (4/4) | Analyze results. Build a decision framework (not just numbers). Document: "Given X constraints, use Y approach because Z" |
| 105 | Jul 22 (Wed) | Review & consolidation | Review all Phase 7 material. Create your personal eval playbook: how to evaluate any new AI feature |

> **Enrichment for Phase 7:** (1) Anthropic "Designing AI-resistant technical evaluations" blog. (2) DL.AI "Building and Evaluating Data Agents." (3) LMSYS Chatbot Arena (chat.lmsys.org) — try judging models yourself. (4) Braintrust `autoevals` library for pre-built scoring.

**Checkpoint:** You can design rigorous evaluations, run experiments, and make data-driven go/no-go decisions.

---

## Phase 8: Production Systems & Deployment

### Week 16 (Jul 23 - Jul 29)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 106 | Jul 23 (Thu) | Quantization theory | GPTQ, AWQ, GGUF — how each works. Precision vs quality tradeoffs. When to use which format |
| 107 | Jul 24 (Fri) | Quantization hands-on | Quantize a model yourself. Compare FP16 vs INT8 vs INT4 output quality. Measure speedup |
| 108 | Jul 25 (Sat) | Inference: KV caching deep dive | KV cache mechanics (you saw this in Phase 3). Now: continuous batching, speculative decoding details |
| 109 | Jul 26 (Sun) | Cost modeling | $/1M tokens calculation for different setups. Cloud vs on-prem tradeoffs. Build a cost spreadsheet |
| 110 | Jul 27 (Mon) | vLLM hands-on | Install vLLM. Serve a model. Benchmark throughput and latency. Test with concurrent requests |
| 111 | Jul 28 (Tue) | llama.cpp / ollama | Serve the same model with ollama. Compare: throughput, latency, ease of use vs vLLM |
| 112 | Jul 29 (Wed) | TGI (Text Generation Inference) | HF's production server. Compare with vLLM and ollama. Understand the tradeoffs between all three |

### Week 17 (Jul 30 - Aug 5)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 113 | Jul 30 (Thu) | MLOps fundamentals (1/2) | Model versioning (HF Hub, MLflow). Prompt versioning and management strategies. Start: MLOps Zoomcamp Module 1 (experiment tracking with MLflow) |
| 114 | Jul 31 (Fri) | MLOps fundamentals (2/2) | Monitoring: latency, error rates, output quality drift detection (Evidently AI). CI/CD for ML: testing model outputs. Reference: huyenchip.com/mlops |
| 115 | Aug 1 (Sat) | Human-in-the-loop design | When to route to humans (confidence thresholds). Feedback loops: corrections → retraining data. Active learning. Read: Anthropic "Effective harnesses for long-running agents" blog |
| 116 | Aug 2 (Sun) | **Project:** deploy service (1/4) | Design API architecture. Choose your model (fine-tuned from Phase 4 or RAG from Phase 5) |
| 117 | Aug 3 (Mon) | **Project:** deploy (2/4) | Quantize model. Set up serving with vLLM or TGI. Build API layer (FastAPI or similar) |
| 118 | Aug 4 (Tue) | **Project:** deploy (3/4) | Add monitoring, logging, error handling. Implement human-in-the-loop fallback for low-confidence |
| 119 | Aug 5 (Wed) | **Project:** deploy (4/4) | Load test. Measure latency/throughput under load. Document deployment architecture and decisions |

> **Enrichment for Phase 8:** (1) MLOps Zoomcamp full course (github.com/DataTalksClub/mlops-zoomcamp) — go deeper on MLflow, Docker, monitoring. (2) Chip Huyen "Designing Machine Learning Systems" book — read chapters on data distribution shifts, monitoring, continual learning. (3) Stanford CS 329S ML Systems Design lectures. (4) DL.AI "Semantic Caching for AI Agents" — production latency optimization. (5) UC Berkeley Agentic AI course (rdi.berkeley.edu/agentic-ai/f25) — production agent infrastructure. (6) Evidently AI ML Observability Course (7 weeks, free) for monitoring deep dive.

**Checkpoint:** You can take any HF model → quantize → serve → monitor → collect feedback → improve. The full loop.

---

## Capstone: Commerce AI System

### Week 18-19 (Aug 6 - Aug 17)

| Day | Date | Topic | What to do |
|-----|------|-------|------------|
| 120 | Aug 6 (Thu) | Architecture design | Design the full system: multimodal product pipeline. Define components, data flow, API contracts |
| 121 | Aug 7 (Fri) | Data pipeline | Build ingestion: product images + descriptions + structured specs. Preprocessing, embedding |
| 122 | Aug 8 (Sat) | RAG layer | Product catalog RAG: vector store setup, hybrid search, metadata filtering for commerce |
| 123 | Aug 9 (Sun) | Multimodal integration | Connect VLM for image understanding. Integrate with text + structured data pipeline |
| 124 | Aug 10 (Mon) | Fine-tuned classifier | Apply your fine-tuned model for category/quality assessment. Connect to the pipeline |
| 125 | Aug 11 (Tue) | Human-in-the-loop | Build the HITL component: confidence routing, human review queue, feedback collection |
| 126 | Aug 12 (Wed) | Evaluation framework | Build offline eval: test dataset, automated metrics, comparison baselines |
| 127 | Aug 13 (Thu) | Deploy & serve | Quantize, serve with vLLM/TGI, API layer, monitoring |
| 128 | Aug 14 (Fri) | Integration testing | End-to-end testing. Fix issues. Load test. Edge cases |
| 129 | Aug 15 (Sat) | Documentation & portfolio | Write up: architecture decisions, tradeoffs, evaluation results. README for GitHub |
| 130 | Aug 16 (Sun) | Polish & demo | Build a simple demo (Gradio/Streamlit). Record a walkthrough or write a blog post |
| 131 | Aug 17 (Mon) | Final review | Review everything. Update resume. Prepare to discuss every decision in an interview |

---

## Calendar Summary

| Month | Dates | Phases | Days |
|-------|-------|--------|------|
| Apr | 9-30 | Phase 1 + Phase 2 start | 22 |
| May | 1-31 | Phase 2 finish + Phase 3 + Phase 4 start | 31 |
| Jun | 1-30 | Phase 4 finish + Phase 5 + Phase 6 start | 30 |
| Jul | 1-31 | Phase 6 finish + Phase 7 + Phase 8 start | 31 |
| Aug | 1-17 | Phase 8 finish + Capstone | 17 |

## Phase Boundaries

| Phase | Start | End | Days |
|-------|-------|-----|------|
| 1. NN Foundations | Apr 9 | Apr 24 | 16 |
| 2. Transformers | Apr 25 | May 13 | 19 |
| 3. LLMs at Scale | May 14 | May 25 | 12 |
| 4. Fine-Tuning | May 26 | Jun 11 | 17 |
| 5. RAG & Retrieval | Jun 12 | Jun 29 | 18 |
| 6. Multimodal | Jun 30 | Jul 9 | 10 |
| 7. Evaluation | Jul 10 | Jul 22 | 13 |
| 8. Production | Jul 23 | Aug 5 | 14 |
| C. Capstone | Aug 6 | Aug 17 | 12 |

---

## Rules for Yourself

1. **2 hours means 2 hours.** Timer on. No "just 10 more minutes" — consistency beats intensity.
2. **If you finish early,** review yesterday's notes. Don't jump ahead.
3. **If you fall behind,** skip the review/consolidation days, not the build days.
4. **Every project must produce a GitHub commit.** Not "I understood it" — code or it didn't happen.
5. **Weekend = normal days.** The schedule already accounts for 7 days/week.
6. **If stuck for >30min,** move on and revisit tomorrow with fresh eyes.

---

## Enrichment Resources Quick Reference

Use these when you finish a day early, want to go deeper, or need a different perspective.
All free unless marked otherwise.

### Courses & MOOCs

| Resource | Phases | Format | URL |
|----------|--------|--------|-----|
| DeepLearning.AI Short Courses (11+ relevant) | P2-P8 | Video + code | deeplearning.ai/courses |
| UC Berkeley LLM Agents MOOC | P5, P8 | Video lectures | llmagents-learning.org/f24 |
| UC Berkeley Advanced LLM Agents | P5, P8 | Video lectures | agenticai-learning.org/sp25 |
| UC Berkeley Agentic AI | P8 | Video lectures | rdi.berkeley.edu/agentic-ai/f25 |
| MLOps Zoomcamp | P8 | Video + projects | github.com/DataTalksClub/mlops-zoomcamp |
| Evidently AI ML Observability | P8 | 7-week course | evidentlyai.com |
| Activeloop GenAI Courses | P5 | 68+ lessons | learn.activeloop.ai |
| W&B AI Academy | P7-P8 | Video + code | wandb.ai/site/courses |

### Engineering Blogs (must-reads marked with *)

| Post | Phase | URL |
|------|-------|-----|
| * Building effective agents | P5 | anthropic.com/research/building-effective-agents |
| * Effective context engineering | P5 | anthropic.com/engineering/effective-context-engineering-for-ai-agents |
| * Writing effective tools for agents | P5-P8 | anthropic.com/engineering/writing-tools-for-agents |
| * Contextual Retrieval | P5 | anthropic.com/engineering/contextual-retrieval |
| The "think" tool | P5 | anthropic.com/engineering/claude-think-tool |
| * Demystifying evals for AI agents | P7 | anthropic.com/engineering/demystifying-evals-for-ai-agents |
| AI-resistant technical evaluations | P7 | anthropic.com/engineering/AI-resistant-technical-evaluations |
| * Effective harnesses for long-running agents | P8 | anthropic.com/engineering/effective-harnesses-for-long-running-agents |
| Advanced tool use | P8 | anthropic.com/engineering/advanced-tool-use |
| Sebastian Raschka: 4 LLM Eval Approaches | P7 | magazine.sebastianraschka.com/p/llm-evaluation-4-approaches |

### Hands-On Notebooks & Cookbooks

| Resource | Phases | URL |
|----------|--------|-----|
| Claude Cookbooks | P5-P6 | github.com/anthropics/claude-cookbooks |
| OpenAI Cookbook | P5 | cookbook.openai.com |
| LlamaIndex Resources | P5 | developers.llamaindex.ai |
| Prompt Engineering Guide | P5-P8 | promptingguide.ai |

### Books

| Book | Phases | Notes |
|------|--------|-------|
| Chip Huyen — "Designing ML Systems" | P8 | The definitive production ML book |
| Dive into Deep Learning (d2l.ai) | P1-P2 | Free, runnable code reference |

### Multimodal References

| Resource | Phase | URL |
|----------|-------|-----|
| VLM Architecture Guide (Rohit Bandaru) | P6 | rohitbandaru.github.io/blog/Vision-Language-Models |
| BentoML Open-Source VLM Guide | P6 | bentoml.com/blog/multimodal-ai-guide |
| DeepLearning.AI Multimodal Search & RAG | P5-P6 | deeplearning.ai/short-courses/building-multimodal-search-and-rag |

---

*Generated: 2026-04-09*
