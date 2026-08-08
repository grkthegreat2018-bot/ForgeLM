# Self-Play Architecture Research — 2026-08-07

## Research Summary

### 1. Model-Generated Questions (Infinite Curriculum)

**Best papers found:**

1. **Absolute Zero Reasoner (AZR)** [NeurIPS 2025] — Single model proposes AND solves code reasoning tasks. Uses Python executor as verifier. Three task types: deduction, abduction, induction. Zero external data. SOTA on coding+math. Code: github.com/LeapLabTHU/Absolute-Zero-Reasoner

2. **SQLM (Self-Questioning Language Models)** [2025] — Asymmetric self-play: proposer generates questions, solver solves. Proposer reward = "not too easy, not too hard" (based on solver agreement variance). For coding: proposer generates unit tests as verification. Works on multiplication, algebra, Codeforces.

3. **SPICE (Self-Play In Corpus Environments)** [Meta FAIR, 2025] — Challenger mines documents → generates QA pairs. Reasoner solves without seeing source document. Information asymmetry prevents collapse. +8.9% math, +9.8% general reasoning. Corpus grounding is KEY — prevents hallucination loops.

4. **SOAR** [2026] — Teacher proposes problems, student solves. Teacher rewarded by student's IMPROVEMENT on hard problems (not intrinsic reward). Grounded rewards > intrinsic rewards. Escapes plateaus at 0/128 success rate.

5. **TTCS** [2026] — Test-time curriculum synthesis. Synthesizer generates progressively harder variants of test questions. Solver trains on variants. Co-evolution loop.

6. **ANCORA** [2026] — UCB-guided Curriculum DAG. Proposer creates verified specs, solver solves. Two-level group-relative updates. Dafny2Verus 26.6% → 81.5%.

7. **PSV (Propose, Solve, Verify)** [2025] — Formal verification for self-play in code generation. Proposer generates formal specs, solver implements, verifier checks. 9.6x improvement over expert iteration.

**Key insight for ForgeAI:** Use the model itself to generate coding tasks with unit tests. The Python executor is the verifier (we already have this!). The proposer reward = "Goldilocks difficulty" (solver succeeds ~50% of the time). This creates an infinite curriculum that adapts to the model's current capability.

### 2. LLM-as-Judge for Expert Orchestration

**Best papers:**

1. **Expert Orchestration (EO)** [2025] — Position paper. Judge models assess expert capabilities. Router directs queries to best specialist. Democratizes LLM advancement. Judge + Router = transparent, controllable.

2. **RouteMoA** [ACL 2026] — Lightweight scorer pre-screens queries → narrows expert candidates WITHOUT inference. Mixture of judges refines. 89.8% cost reduction, 63.6% latency reduction.

3. **LLM-as-Scheduler (LAS)** [ACL 2026] — Two-stage cascade: lightweight gate evaluates output, LLM scheduler routes. 43% token reduction, 36% latency reduction. Only 1.4pp accuracy drop.

4. **EvoRoute** [ACL 2026] — Self-evolving routing. Builds experience knowledge base. Dynamically selects Pareto-optimal LLM per step. 80% cost reduction, 70% latency reduction.

5. **Uno-Orchestra** [2026] — Unified orchestration policy. Selectively decomposes tasks, dispatches subtasks to (model, primitive) pairs. RL-learned. 77% macro pass@1, 10x lower cost.

6. **RACER** [2026] — Robust Adaptive Cost-Efficient Routing for LLM-as-Judge. Routes between reasoning/non-reasoning judges based on task. Distributionally robust optimization.

**Key insight for ForgeAI:** The base model (no experts) acts as judge+router. It classifies the query → selects expert → evaluates expert output → decides to accept/retry/kill. This is a tool-use pattern: `call_expert(topic)`, `kill_expert(topic)`, `improve_expert(topic, feedback)`.

### 3. Self-Referential Improvement (Train on Own Code)

**Best papers:**

1. **SICA (Self-Improving Coding Agent)** [2025] — Agent edits its OWN codebase to improve itself. 17% → 53% on SWE-Bench Verified. No distinction between meta-agent and target agent.

2. **Gödel Agent** [ACL 2025] — Self-referential framework. LLM dynamically modifies its own logic/behavior via prompting. Continuous self-improvement. Surpasses hand-crafted agents.

3. **Darwin Gödel Machine (DGM)** [2025] — Open-ended evolution. Iteratively modifies own code. Empirically validates each change. Code archive = knowledge base.

4. **Bilevel Autoresearch** [2026] — Outer loop optimizes inner loop by generating new Python mechanisms at runtime. 5x improvement on GPT pretraining benchmark. Same LLM for both loops.

5. **WARNING: Recursive Self-Training Collapse** [2026] — AI reviewing its own code degenerates to rubber-stamping. AI-self-gate filters lose effectiveness over iterations. Need EXOGENOUS verification (tests, compilers, human review) not model-coupled self-review.

**Key insight for ForgeAI:** Train on own source code is viable BUT requires exogenous verification (unit tests, execution results). The model can propose improvements to its own training code, but changes must be validated by running the actual training pipeline and measuring benchmark results — NOT by the model judging its own code.

### 4. Architecture Design for ForgeAI

Combining all research into a concrete plan:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LAYER                        │
│  (Base model — no experts loaded)                           │
│                                                             │
│  1. Classify query → topic                                  │
│  2. Route to expert (or solve directly if confident)        │
│  3. Judge expert output (accept/retry/kill)                 │
│  4. If retry: generate feedback for expert improvement      │
│                                                             │
│  Tools: call_expert(topic, query) → response                │
│         kill_expert(topic)                                  │
│         improve_expert(topic, solutions)                    │
│         generate_task(topic, difficulty) → task+tests       │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Expert A │    │ Expert B │    │ Expert N │
    │ (python) │    │ (math)   │    │ (logic)  │
    │ AirMoE   │    │ AirMoE   │    │ AirMoE   │
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           ▼               ▼               ▼
    ┌──────────────────────────────────────────┐
    │         INFINITE CURRICULUM ENGINE         │
    │                                          │
    │  Proposer: generates tasks + unit tests  │
    │  Verifier: Python executor checks tests  │
    │  Difficulty reward: ~50% solver success  │
    │                                          │
    │  Tasks stored in curriculum queue        │
    │  Difficulty adapts to solver capability  │
    └──────────────────────────────────────────┘
```

### 5. Implementation Priorities

**Phase 1: Infinite Curriculum (highest impact)**
- Implement AZR-style task proposer: model generates coding tasks + unit tests
- Python executor verifies (we already have this infrastructure)
- Goldilocks difficulty reward: proposer rewarded when solver succeeds 40-60%
- Task queue: store generated tasks, replay for training
- This replaces the fixed GoalTaskGenerator with infinite, adaptive tasks

**Phase 2: Orchestrator/Judge**
- Base model classifies query → routes to best expert
- Base model judges expert output: "is this correct? is this better than what I'd produce?"
- Tool-use interface: call_expert, kill_expert, improve_expert
- Judge training: fine-tune base model on (query, expert_output, verdict) triples

**Phase 3: Self-Referential Improvement**
- Model reads its own training code (self_play_expert_training.py, etc.)
- Proposes improvements (new reward functions, new task types, new architectures)
- Changes validated by running actual training + benchmark measurement
- Exogenous verification only — no model-coupled self-review
- Archive of validated improvements (Darwin Gödel Machine style)

**Phase 4: Continuous Loop**
- Train experts on infinite curriculum
- Orchestrator learns to route better
- Model proposes new task types as it masters existing ones
- Periodically: model reviews own code, proposes improvements, validates via benchmarks
