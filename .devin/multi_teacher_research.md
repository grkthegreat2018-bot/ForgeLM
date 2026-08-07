# Web Search Results for "small LLM coherence improvement techniques beyond pretraining distillation instruction tuning"

## 1. Symbolic Chain-of-Thought Distillation: Small Models Can Also “Think” Step-by-Step
URL: https://arxiv.org/html/2306.14050v1

Chain-of-thought prompting (e.g., “Let’s think step-by-step") primes large language models to verbalize rationalization for their predictions. While chain-of-thought can lead to dramatic performance gains, benefits appear to emerge only for sufficiently large models (beyond 50B parameters). We show that orders-of-magnitude smaller models (125M—1.3B parameters) can still benefit from chain-of-thought prompting. To achieve this, we introduce Symbolic Chain-of-Thought Distillation (SCoTD), a method to train a smaller student model on rationalizations sampled from a significantly larger teacher model. Experiments across several commonsense benchmarks show that: 1) SCoTD enhances the performance of the student model in both supervised and few-shot settings, and especially for challenge sets; 2) sampling many reasoning chains per instance from the teacher is paramount; and 3) after distillation, student chain-of-thoughts are judged by humans as comparable to the teacher, despite orders of magnitude fewer parameters. We test several hypotheses regarding what properties of chain-of-thought samples are important, e.g., diversity vs. teacher likelihood vs. open-endedness. We release our corpus of chain-of-thought samples and code.
...
However, chain-of-thought prompting has only been shown to be beneficial for models of sufficient scale (e.g., with more than 60B parameters Wei et al. (2022b)). In this work, we study whether small language models can be “taught" the capacity for chain-of-thought reasoning by larger language models. We adopt a simple strategy, which we call Symbolic Chain-of-thought Distillation (SCoTD): first, we sample chain-of-thought rationales from large language model given (unlabeled) input instances from a dataset; then, we train a smaller language model to predict the sampled rationale and sampled label. This process follows the “symbolic knowledge distillation” paradigm as in West et al. (2022), wherein corpora are sampled from a larger language model...

## 2. Bridging the Efficiency-Consistency Gap: Enabling Logical Coherence in Resource-Constrained Language Models
URL: https://openreview.net/pdf/dfa1dd616ee8511a9e6e4c78c35232ff2d8b3f86.pdf

tinguishes factual consistency (external truth align- 128 
ment) from logical consistency (formal reasoning 129 
adherence), with Liu et al. (2025) establishing met- 130 
rics for transitivity, commutativity, and negation 131 
invariance, while Ghosh et al. (2025) demonstrate 132 
large models require targeted supervision for con- 133 
sistent logical operators. 134 
2.2 Approaches to Consistency Enhancement 135 
Existing consistency enhancement methods fall 136 
into two fundamental categories with dramatically 137 
different computational profiles: inference-time 138 
methods that verify consistency through post-hoc 139 
checking, and training-time methods that embed 140 
consistency into model weights. 141 
Inference-time methods offer flexibility at the 142 
cost of computational overhead. Chain-of-thought 143 
prompting (Wei et al., 2022) generates intermediate 144 
reasoning steps requiring 2-5 additional forward 145 
passes per query. Self-correction approaches itera- 146 
tively refine outputs through 3-10 verification cy- 147 
cles. Metacognitive frameworks like SPOC (Zhao 148 
et al., 2025) interleave solution generation and ver- 149 
ification, compounding latency. While effective 150 
for large models with substantial compute budgets, 151 
these methods fundamentally preclude deployment 152 
in resource-constrained scenarios where base infer- 153 
ence latency already approaches acceptable thresh- 154 
olds. 155 
Training-time methods embed consistency dur- 156 
ing model training, incurring costs once rather 157 
than per query. Process-level supervision meth- 158 
ods like S²R (Ma et al., 2025) combine supervised 159 
fine-tuning with reinforcement learning to reward 160 
valid reasoning steps. Consistency Reward Models 161 
(CRMs) train dedicated scorers for logical coher- 162 
ence (Leung and Wang, 2025). Neuro-symbolic 163 
integration approaches, particularly LoCo-LMs 164 
(Calanzone et al., 2024), use differentiable loss 165 
functions to penalize...

## 3. Enhancing Reasoning Abilities of Small LLMs with Cognitive Alignment
URL: https://p.rst.im/q/aclanthology.org/2025.emnlp-main.377.pdf

when solving problems compared to their larger counterparts, as illustrated in Figure 1. Similar find ings have also been presented in (Li et al., 2022; Zhang et al., 2024; Hu et al., 2024; Li et al., 2024). This phenomenon indicates that direct distillation of CoTs from larger models can sometimes be in effective due to the large capacity gap. Thus, a natural question arises: How can we improve the reasoning abilities of smaller LRMs in a way that is aligned with their own cognitive capacity? In this paper, we introduce the “Critique Rethink-Verify” (CRV) system, a novel approach to enhance the reasoning capabilities of smaller models. CRV leverages multiple LLM agents, each with specialized functions working in synergy: (i) critiquing CoT rationales by considering the cog nitive limits of smaller LRMs, (ii) rethinking and refining these CoTs, integrating the feedback from previous critiques, and (iii) verifying the accuracy and validity of the refined reasoning paths. Extend ing the Direct Preference Optimization (DPO) tech nique (Rafailov et al., 2023), we further propose the Cognitive Preference Optimization (CogPO) al gorithm to align the reasoning process with the cog nitive capacities of smaller LRMs, building upon the CRV system. Ultimately, the reasoning perfor mance of smaller models can be improved effec tively. We evaluate the effectiveness of our approach on several challenging reasoning benchmarks that are difficult for models with limited parameter sizes, such as AIME 2024, MATH-500 (Lightman et al., 2023), GPQA-Diamond (Rein et al., 2023), and LiveCodeBench. The results indicate that the small LRMs trained using the CRV+CogPO framework achieve outstanding reasoning performance. In summary, our major contributions are:
...
Figure 2: Overview of our CRV+CogPO framework
...
(2) CogPO
...
backbone; however, any LLMs with sufficient capabilities can serve as agents as well.
...
and many others. 2.3 Alignment Training To effectively train LLMs, a reinforce...

## 4. 
URL: https://openreview.net/attachment?id=K1X49CM6GK&name=pdf

further introduce an in-context example generator and a teacher-forcing
...
Chain-of-Thought strategy to ensure that the rationales are accurate and
...
However, existing methods suffer from two major drawbacks: (1) Limited Knowledge
...
To solve these issues, we propose TINYLLM, a paradigm that facilitates the learning of
...
a small student LLM by distilling knowledge from multiple large teacher LLMs with
...
rationale guidance. Specifically, TINYLLM mitigates the limited knowledge diversity issue
...
by involving multiple teacher models as co-advisors, which introduces a richer, varied
...
knowledge source for the student to learn from. To fully exploit each teacher model and
...
mitigate the lack of rich contextual information problem, TINYLLM asks the teacher for
...
credible rationales to support the answers, thereby providing the student with a deeper
...
understanding of the problem-solving process. By learning from multiple teachers, the
...
student model can inherit a broader range of skills and knowledge, leading to better
...
generalization capabilities. In addition, to ensure the rationales are grounded in contextually
...
appropriate scenarios and reflect the true underlying reasoning procedure, TINYLLM
...
features an in-context example generator and a teacher-forcing Chain-of-Thought strategy,
...
making the teachers understand the task through demonstrations and therefore generate
...
the accurate rationales.
...
• TINYLLM encompasses several innovative designs including an in-context example
...
generator, a teacher-forcing Chain-of-Thought strategy, and a joint learning objective
...
from various teachers.
...
test-time, not fully addressing deployment challenges. In this work, we propose a multi-task
...
learning paradigm with superior chain-of-thought reasoning capabilities, avoiding the
...
addition, the employment of the Chain-of-Thought paradigm has facilitated the generation
...
of deliberative reasoning samples from the teacher models (Ho e...

## 5. Training Small Reasoning LLMs with Cognitive Preference Alignment
URL: https://arxiv.org/html/2504.09802v1

The reasoning capabilities of large language models (LLMs), such as OpenAI’s o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need to explore strategies to train effective reasoning LLMs with far fewer parameters. A critical challenge is that smaller models have different capacities and cognitive trajectories than their larger counterparts. Hence, direct distillation of chain-of-thought (CoT) results from large LLMs to smaller ones can be sometimes ineffective and requires a huge amount of annotated data. In this paper, we introduce a novel framework called Critique-Rethink-Verify (CRV), designed for training smaller yet powerful reasoning LLMs. Our CRV framework consists of multiple LLM agents, each specializing in unique abilities: (i) critiquing the CoTs according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. We further propose the cognitive preference optimization (CogPO) algorithm to enhance the reasoning abilities of smaller models by aligning thoughts of these models with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of CRV and CogPO, which outperforms other training methods by a large margin. 111Source codes, datasets and models will be released upon paper acceptance.
...
A straightforward approach to address this challenge is the direct distillation of Chain-of-Thought (CoT) outputs Wei et al. (2022a) or other deep thoughts (such as Tree-of-Thought Yao et al. (2023b)) from larger LLMs to smaller models. This technique is widely applied to improve the capacities of smaller LLMs Hsieh et al. (2023); Shridhar et al. (2022); Li et al. (2023); Yue et al. (2024). However, smaller LLMs333In this work, we regard smaller LLMs as decoder-only language models typi...
