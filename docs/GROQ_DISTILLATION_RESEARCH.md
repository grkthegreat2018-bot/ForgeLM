# Groq API — Distillation & Training Data Research

## Verdict: YES, distillation is allowed — but ONLY with Apache 2.0 models

### The Key Insight

Groq is an **inference provider**, not a model developer. They serve open-weight models
on their LPU hardware. Groq's own terms do NOT restrict how you use outputs — the
**underlying model license** governs distillation rights.

> "The mental model that trips people up: Groq does not make the models it serves.
> It runs open-weight models from Meta, Mistral, Google, Alibaba and others on its
> own LPU hardware. So 'using Groq' means using, say, Llama — and Llama's own
> license governs what you may do with the model and its outputs."
> — atlas.lb-product.com

Groq also does NOT train on your data (confirmed by multiple sources). Zero Data
Retention (ZDR) is self-serve in the console.

---

## Model-by-Model License Analysis

### ✅ SAFE for Distillation (Apache 2.0)

| Model | Params | Context | Speed | Input $/1M | Output $/1M | Free RPM | Free TPM | Free RPD |
|-------|--------|---------|-------|------------|-------------|----------|----------|----------|
| `qwen/qwen3-32b` | 32B | 131K | ~662 t/s | $0.29 | $0.59 | 60 | 6,000 | 1,000 |
| `openai/gpt-oss-120b` | 120B | 131K | ~500 t/s | $0.15 | $0.60 | 30 | 8,000 | 1,000 |
| `openai/gpt-oss-20b` | 20B | 131K | ~1,000 t/s | $0.075 | $0.30 | 30 | 8,000 | 1,000 |

Apache 2.0 grants "perpetual, worldwide, non-exclusive, no-charge, royalty-free,
irrevocable copyright license to reproduce, prepare Derivative Works of, publicly
display, publicly perform, sublicense, and distribute the Work and such Derivative
Works." No restrictions on using outputs to train other models.

### ❌ UNSAFE for Distillation (Llama Community License)

| Model | License Clause |
|-------|---------------|
| `llama-3.1-8b-instant` | "You will not use the Llama Materials or any output or results |
| `llama-3.3-70b-versatile` | of the Llama Materials to improve any other large language model |
| `meta-llama/llama-4-scout-17b-16e-instruct` | (excluding Meta Llama 3 or derivative works thereof)." |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | |
| `deepseek-r1-distill-llama-70b` | (Llama derivative — Llama license applies) |

**ForgeAI is LFM2.5-based (Liquid AI), NOT a Llama derivative.** Using Llama outputs
to train ForgeAI would VIOLATE the Llama Community License.

### ⚠️ RISKY (Gemma Terms of Use)

| Model | Issue |
|-------|-------|
| `gemma2-9b-it` | "Model Derivatives" includes "any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma, to that model in order to cause that model to perform similarly to Gemma, including distillation methods that use... generation of synthetic data Outputs by Gemma for training that model." |

Outputs themselves are NOT derivatives, but a model trained on them to replicate
Gemma's performance becomes a "Model Derivative" subject to Gemma Terms (including
Google's right to remotely restrict usage). Avoid for distillation.

---

## Free Tier Rate Limits (Per Organization, Not Per API Key)

Rate limits apply at the **organization level** — multiple API keys do NOT multiply
your quota. Prompt caching tokens do NOT count toward TPM limits.

| Model | RPM | TPM | RPD | Notes |
|-------|-----|-----|-----|-------|
| `qwen/qwen3-32b` | 60 | 6,000 | 1,000 | **2x RPM** — best for parallel requests |
| `openai/gpt-oss-20b` | 30 | 8,000 | 1,000 | Fastest (1000 t/s) — best for throughput |
| `openai/gpt-oss-120b` | 30 | 8,000 | 1,000 | Strongest reasoning — best for complex tasks |
| `llama-3.1-8b-instant` | 30 | 6,000 | 14,400 | Highest RPD but ❌ license-blocked |
| `llama-3.3-70b-versatile` | 30 | 12,000 | 1,000 | ❌ license-blocked |

**Daily free-tier capacity (Apache 2.0 models only):**
- Qwen3-32B: 1,000 requests × ~500 output tokens = ~500K tokens/day
- gpt-oss-20b: 1,000 requests × ~500 output tokens = ~500K tokens/day
- gpt-oss-120b: 1,000 requests × ~500 output tokens = ~500K tokens/day
- **Combined: ~1.5M tokens/day of free training data**

---

## Maximizing Groq API Usage for ForgeAI

### 1. Multi-Model Teacher Ensemble
Use all 3 Apache 2.0 models in parallel (rate limits are per-model):
- **Qwen3-32B**: Primary teacher for reasoning/code tasks (60 RPM, strong thinking mode)
- **gpt-oss-120b**: Complex multi-step problems (131K context, high reasoning effort)
- **gpt-oss-20b**: High-throughput simple tasks (~1,000 t/s, lowest cost)

### 2. Batch API for Bulk Generation
Groq's Batch API gives ~25% discount on on-demand pricing. Submit large JSONL
files of prompts, get results asynchronously. Ideal for generating thousands of
training examples overnight.

### 3. Prompt Caching
Cached input prefixes don't count toward TPM. Use long system prompts with
task-specific suffixes — the system prompt is cached, only the suffix counts.

### 4. Reasoning Mode Control (Qwen3)
Qwen3 supports `reasoning_effort: none|default|low|medium|high`:
- Use `high` for complex reasoning (generates rich CoT traces for distillation)
- Use `none` for simple tasks (faster, cheaper, no reasoning tokens)
- Use `default` for general purpose

### 5. Structured Outputs
Use `response_format: { type: "json_schema", json_schema: {...} }` to force
structured JSON output — ideal for generating (prompt, solution, test_cases)
triples for self-play training data.

### 6. Temperature Diversification
Run the same prompt at temperatures 0.2, 0.5, 0.8, 1.2 to generate diverse
completions for GRPO group sampling (need >= 2 completions per prompt for
advantage computation).

### 7. Distillation Pipeline Integration
```
Groq API (Qwen3/gpt-oss) → generate (prompt, solution, test_cases) triples
    ↓
Sandbox verification (self_play_sandbox.py) → filter to correct-only
    ↓
ReplayBuffer (replay_buffer.py) → store golden trajectories
    ↓
GRPOTrainer (grpo_trainer.py) → inject as golden replays during training
    ↓
GoalScorer (goal_scorer.py) → score with minimalism_active=True (GRPO-λ)
```

### 8. Cost Projection (Paid Tier)
At Developer tier rates, generating 1M training tokens:
- gpt-oss-20b: $0.075 input + $0.30 output ≈ $0.38 per 1M tokens
- Qwen3-32B: $0.29 input + $0.59 output ≈ $0.88 per 1M tokens
- gpt-oss-120b: $0.15 input + $0.60 output ≈ $0.75 per 1M tokens

10M tokens of training data (enough for a full SFT pass) ≈ $4-9 on gpt-oss-20b.

---

## API Integration Notes

- Endpoint: `POST https://api.groq.com/openai/v1/chat/completions`
- OpenAI-compatible API (use `openai` Python client with `base_url` override)
- `n` parameter must be 1 (no batch sampling in single request — use multiple calls)
- `logprobs` not supported on any model
- `seed` parameter available for reproducibility (best-effort, not guaranteed)
- Streaming supported via SSE
- `reasoning_format: "raw"` exposes full CoT tokens (useful for distillation)

### Python Quick Start
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
response = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_completion_tokens=2048,
    reasoning_effort="high",  # Qwen3 only
)
```

---

## Sources
- Groq API Reference: https://console.groq.com/docs/api-reference
- Groq Terms: https://groq.com/groq-service-specific-terms
- Groq Data Retention: https://meetily.ai/llm-privacy/groq
- Qwen3 License (Apache 2.0): https://huggingface.co/Qwen/Qwen3-32B/blob/main/LICENSE
- gpt-oss License (Apache 2.0): https://github.com/openai/gpt-oss/blob/main/LICENSE
- Llama License restriction: https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use/raw/main/LICENSE
- Gemma Terms: https://ai.google.dev/gemma/terms
- Groq rate limits: https://www.eesel.ai/blog/groq-pricing
- Groq pricing: https://www.cloudzero.com/blog/groq-pricing/

---

## Additional Free-Tier Providers (All OpenAI-Compatible API)

### DeepSeek (MIT License — Best for Distillation)
- **License**: MIT — "Distill & commercialize freely!" (explicitly allowed)
- **Models**: `deepseek-reasoner` (R1, shows CoT), `deepseek-chat` (V3)
- **Free tier**: Generous daily limits, no credit card
- **Pricing**: $0.55/M input, $2.19/M output (reasoner)
- **API**: `https://api.deepseek.com/v1`, env: `DEEPSEEK_API_KEY`
- **Why it's great**: R1 reasoning traces are gold for CoT distillation. MIT
  license is the most permissive — no restrictions at all on output use.

### Cerebras ($5 Free Credit — Fastest Inference)
- **License**: Model-dependent (gpt-oss = Apache 2.0)
- **Models**: `gpt-oss-120b` (~3000 t/s — fastest available)
- **Free tier**: $5 credit, expires 30 days, requires verified payment method
- **API**: `https://api.cerebras.ai/v1`, env: `CEREBRAS_API_KEY`
- **Why it's great**: 20x faster than OpenAI. Best for high-throughput bulk
  generation when you need to generate thousands of samples quickly.

### Hyperbolic ($10 Free Credit)
- **License**: Model-dependent (gpt-oss = Apache 2.0, DeepSeek = MIT)
- **Models**: gpt-oss-120b ($0.30/M), gpt-oss-20b ($0.04/M), DeepSeek V3/R1
- **Free tier**: $10 credit on signup, 60 RPM free tier
- **API**: `https://api.hyperbolic.xyz/v1`, env: `HYPERBOLIC_API_KEY`
- **Why it's great**: Cheapest gpt-oss-20b at $0.04/M tokens. $10 free goes far.

### OpenRouter (Free Tier — 20+ Free Models)
- **License**: Model-dependent (check each model)
- **Free models**: Qwen3-235B (Apache 2.0), gpt-oss-120b (Apache 2.0), others
- **Free tier**: 50 RPD (1000 RPD with $10 purchase), 20 RPM
- **API**: `https://openrouter.ai/api/v1`, env: `OPENROUTER_API_KEY`
- **Why it's great**: Single API key, 500+ models, auto-routing. Free models
  include the largest MoE models (Qwen3-235B). `:free` suffix for free variants.

### Together AI ($25 Free Credit — Most Free Credit)
- **License**: Model-dependent
- **Free models**: DeepSeek-R1-Distill-Llama-70B (free endpoint)
  - ⚠️ This is a Llama derivative → Llama license applies → ❌ NO distillation
- **Free tier**: $25 credit, no credit card, 68 free models
- **API**: `https://api.together.ai/v1`, env: `TOGETHER_API_KEY`
- **Note**: Most free models are Llama-licensed. Only use Apache/MIT models
  from Together for distillation (Qwen, DeepSeek non-distilled variants).

### Novita AI ($10 Free Credit)
- **License**: Model-dependent
- **Free models**: Qwen2.5-7B, GLM-4-9B (Apache 2.0)
- **Free tier**: $10 voucher for new users
- **API**: OpenAI-compatible, env: `NOVITA_API_KEY`

### Google AI Studio (Gemini API — ❌ AVOID for Distillation)
- **Free tier**: Free input/output tokens, no credit card
- **Problem**: "When you use Unpaid Services, Google uses the content you
  submit and any generated responses to provide, improve, and develop Google
  products and services and machine learning technologies."
- **Paid tier**: Google doesn't use your data — but Gemini is proprietary,
  not open-weight. No explicit distillation permission.
- **Verdict**: Skip. Google trains on your data (free tier) and Gemini's
  license doesn't explicitly permit distillation.

### Mistral (Free Experiment Plan — ⚠️ Opt-Out Required)
- **Free tier**: ~1B tokens/month, no credit card, ~1 req/sec
- **Problem**: "API requests on the free Experiment plan may be used to train
  Mistral's models. You can opt out in the Admin Console."
- **License**: Mistral Non-Production License (Codestral) — not fully open
- **Verdict**: Opt out of training first. Check license per model. Not ideal
  for distillation due to non-production license restrictions.

---

## Combined Free-Tier Capacity

| Provider | Free Credit | Safe Models | Est. Daily Tokens |
|----------|------------|-------------|-------------------|
| Groq | $0 (permanent) | 3 (Apache) | ~1.5M |
| DeepSeek | $0 (permanent) | 2 (MIT) | ~2M |
| Cerebras | $5 (30 days) | 1 (Apache) | ~3M (fastest) |
| Hyperbolic | $10 | 2 (Apache) | ~2M |
| OpenRouter | $0 / $10 | 2+ (Apache) | ~0.5M-1M |
| Together | $25 | limited | ~1M (careful with licenses) |
| **Combined** | ~$40+ | 10+ models | **~10M tokens/day** |

With ~10M tokens/day of free training data, you can generate a full SFT
dataset (10M tokens) in a single day, or continuously feed the ReplayBuffer
for ongoing self-play augmentation.
