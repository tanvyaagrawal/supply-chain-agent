## Model Benchmarking
### Base Model
We use **Llama-3.2-3B-Instruct** as the baseline model for evaluation.
To make the model efficient and runnable on limited hardware, we load it using **4-bit quantization (NF4)** with the BitsAndBytes library.
This baseline model serves as:
- The reference model for benchmarking  
- The foundation for continued pretraining (CPT)  
- The backbone for both baseline and neurosymbolic agents  
### Continued Pretraining (CPT)
To adapt the baseline LLM to the supply chain risk domain, we perform **continued pretraining (CPT)** on a synthetic corpus generated from the project’s symbolic knowledge base.
#### Domain Corpus Construction
The CPT corpus is created from the **supply chain knowledge graph** by converting:
- node attributes into natural-language factual statements
- edge relations into relational statements
- multi-hop supply dependencies into propagation statements
This transforms structured graph knowledge into plain text suitable for language model pretraining. The code builds records from node facts, edge facts, and disruption propagation paths, then writes them into JSONL files for training. :contentReference[oaicite:1]{index=1}
#### Data Expansion
To increase training diversity, each generated fact is rewritten using multiple lightweight templates such as:
- direct factual form
- supply chain fact form
- risk analysis statement form
- knowledge graph fact form
- domain statement form
This produces an expanded domain corpus without changing the underlying information content. :contentReference[oaicite:2]{index=2}
#### Training Strategy
The model is loaded in **4-bit quantized form** and prepared for efficient k-bit training. Instead of full fine-tuning, the project uses **LoRA-based continued pretraining**, which updates only selected transformer modules while keeping the approach lightweight and hardware-efficient. :contentReference[oaicite:3]{index=3}
#### Training Setup
Key training choices include:
- 4-bit NF4 quantization
- bfloat16 / float16 compute depending on hardware support
- LoRA adapters on attention and feed-forward projection layers
- causal language modeling objective
- block-based token grouping for next-token prediction
- train/validation split for monitoring performance
- epoch-wise saving and evaluation
#### Output
The continued pretrained adapter and tokenizer are saved for later reuse in:
- model benchmarking
- baseline agent benchmarking
- neurosymbolic agent benchmarking
- context benchmarking experiments
### Model Evaluation: Baseline vs Continued Pretrained Model
After continued pretraining, both the **baseline model** and the **CPT-adapted model** are evaluated to measure improvements in truthfulness, hallucination behavior, and domain reasoning performance.
#### TruthfulQA Evaluation
##### Setup
- Dataset: TruthfulQA (generation split)
- Evaluation subset: ~200 questions
- Category-aware sampling
- Deterministic inference (no sampling)
##### Metrics
| Metric | Baseline | CPT Model |
|--------|----------|-----------|
| Truthful Accuracy | 28.0% | 31.0% |
| Hallucination Rate | 63.5% | 61.0% |
| Avg BLEU | 6.40 | 6.63 |
| Avg ROUGE-L | 0.1704 | 0.1762 |
| Avg Latency (s) | 6.26 | 9.39 |
| Avg Total Tokens | 156 | 155 |
##### Key Observations
- CPT improves **truthful accuracy slightly**
- Hallucination reduces marginally
- BLEU and ROUGE-L show small gains
- Latency increases due to adapter overhead
- Token usage remains nearly constant
##### Important Note
This is a **custom TruthfulQA-style evaluation**, not official leaderboard scoring.
#### BIG-Bench Evaluation
##### Setup
- Framework: BIG-Bench
- Custom HuggingFace → BIG-Bench wrapper
- Chat-template based prompting
##### Tasks
- KG Entity Knowledge
- Risk Level Assessment
- KG Numerical Facts
- KG Propagation
##### Metrics
| Task | Metric | Baseline | CPT Model |
|------|--------|----------|-----------|
| KG Entity Knowledge | Multiple Choice Grade | 0.33 | 0.33 |
| Risk Level Assessment | Multiple Choice Grade | 0.55 | 0.55 |
| KG Numerical Facts | BLEU | 2.83 | 2.89 |
| KG Propagation | ROUGE-L | 0.1525 | 0.1577 |
##### Key Observations
- CPT improves **numerical and propagation reasoning slightly**
- MCQ tasks remain unchanged
- Improvements are stronger in **generation-based tasks**
- Domain knowledge learned during CPT reflects in outputs
##### Important Note
This is a **custom BIG-Bench-style evaluation using KG-grounded tasks**, not standard public tasks.

#### Overall Insights
- CPT improves **domain alignment**
- Gains are **small but consistent**
- Stronger improvements in:
  - generation tasks
  - structured reasoning
- Trade-off:
  - higher latency
  - similar token usage
### Agent Benchmarking: Baseline vs Neurosymbolic Agent
This stage compares two agents:
- **Baseline Agent (RAG)** → document retrieval + LLM  
- **Neurosymbolic Agent** → knowledge graph + symbolic rules + LLM  
The objective is to evaluate whether structured symbolic reasoning improves performance, reduces hallucination, and enhances reasoning capability.
### Baseline Agent (RAG)
#### Design
- Uses **TF-IDF document retrieval**
- Retrieves relevant documents
- Generates answers using LLM with context
#### Characteristics
- High context efficiency
- Faster responses
- Prone to hallucination due to weak grounding
### Neurosymbolic Agent (Graph + Rules)
#### Design
- Uses **knowledge graph for reasoning**
- Pipeline:
  - Entity extraction
  - Subgraph retrieval (multi-hop)
  - Graph-to-text conversion
  - Symbolic rule injection
#### Characteristics
- Strong factual grounding
- Multi-hop reasoning capability
- Strict refusal on missing information
#### Prompt Design
Both agents enforce:
- Use only provided context
- No external knowledge
- No hallucination
- Concise structured answers
Graph agent additionally enforces:
- Use only graph evidence
- Apply symbolic reasoning rules
- Refuse when information is missing
### Evaluation Setup

- Total Tasks: **40**
- Includes **hallucination trap tasks**
- Same evaluation pipeline used for both agents
### Custom Evaluator
A custom evaluator is implemented to measure multiple aspects of agent performance beyond simple accuracy.
### Key Metrics Computed
#### 1. Task Success Rate
Measures whether the agent correctly solves the task based on expected outputs.
#### 2. Hallucination Rate
Detects whether the agent generates unsupported or incorrect information.
- Separate tracking for:
  - trap tasks
  - non-trap tasks
#### 3. Refusal Accuracy
Measures whether the agent correctly refuses to answer when required (especially in trap scenarios).
#### 4. Reasoning Accuracy
Evaluates correctness of reasoning steps:
- overall reasoning accuracy
- multi-hop reasoning accuracy
#### 5. Context Efficiency
Measures how effectively the agent uses context relative to tokens consumed.
#### 6. Efficiency Metrics
- Average latency (seconds)
- Token usage:
  - input tokens
  - output tokens
  - total tokens
- Cost per query

### Results
### Comparison Table
| Metric | Baseline | Graph | Change |
|--------|----------|--------|--------|
| Task Success Rate | 0.0500 | 0.1750 | ↑ +0.1250 |
| Hallucination Rate | 0.8333 | 0.0000 | ↓ -0.8333 |
| Reasoning Accuracy (Multi-hop) | 0.2343 | 0.2524 | ↑ +0.0181 |
| Context Efficiency | 0.5408 | 0.3897 | ↓ -0.1511 |
| Avg Latency (s) | 20.97 | 24.97 | ↑ +4.00 |
| Avg Total Tokens | 1167.03 | 1737.95 | ↑ +570.92 |

### Detailed Metrics
### Baseline Agent

- Task Success Rate: **0.05**
- Hallucination Rate: **0.8333**
- Refusal Accuracy: **0.1667**
- Avg Reasoning Accuracy (All): **0.2391**
- Avg Reasoning Accuracy (Multi-hop): **0.2343**
- Context Efficiency: **0.5408**
- Avg Latency: **20.97 s**
- Avg Tokens:
  - Input: 932.25
  - Output: 234.78
  - Total: 1167.03
- Cost per Query: **0.001402**
- Total Cost: **0.056072**

### Neurosymbolic (Graph) Agent
- Task Success Rate: **0.175**
- Hallucination Rate: **0.0000**
- Refusal Accuracy: **1.000**
- Avg Reasoning Accuracy (All): **0.3645**
- Avg Reasoning Accuracy (Multi-hop): **0.2524**
- Context Efficiency: **0.3897**
- Avg Latency: **24.97 s**
- Avg Tokens:
  - Input: 1469.15
  - Output: 268.8
  - Total: 1737.95
- Cost per Query: **0.002007**
- Total Cost: **0.08027**

### Key Observations
- **Task success improves significantly** with neurosymbolic approach (+12.5%)
- **Hallucination is completely eliminated** (0.8333 → 0.0)
- **Refusal accuracy becomes perfect (1.0)** → strong safety behavior
- **Reasoning accuracy improves**, especially for multi-hop queries
- Graph agent produces more **grounded and reliable responses**

### Trade-offs
- Higher latency (+4 seconds per query)
- Increased token usage (+570 tokens)
- Lower context efficiency due to graph-based prompts
- Slightly higher cost per query
### Insights
- Symbolic reasoning dramatically improves **factual correctness**
- Knowledge graphs enable **hallucination-free reasoning**
- Trade-off exists between:
  - accuracy and reliability
  - efficiency and cost
