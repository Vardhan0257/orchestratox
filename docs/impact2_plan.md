# OrchestraTox – Impact 2

## Working Title

Cross-Model Transferability of Multi-Agent Prompt Injection Attacks

---

## Research Question

To what extent do prompt-injection attacks transfer across different large language models operating under the same multi-agent orchestration framework?

---

## Motivation

Impact 1 demonstrated that orchestration architecture affects target-action selection rates within a fixed benchmark. Impact 2 investigates whether these attack behaviours are specific to one LLM or generalise across multiple LLMs.

---

## Hypothesis (H1)

Prompt-injection attacks exhibit partial cross-model transferability. Some attack scenarios will consistently succeed across multiple models, while others will be model-specific.

---

## Null Hypothesis (H0)

Attack success is independent of the underlying LLM.

---

## Independent Variable

- LLM used in the orchestration framework

---

## Controlled Variables

- Same 15 human-reviewed scenarios
- Same prompts
- Same target and benign actions
- Same evaluation logic
- Same temperatures
- Same orchestration framework
- Same trial count

---

## Dependent Variable

Target-action selection (Yes/No)

---

## Expected Outputs

- Per-model attack success rates
- Cross-model comparison tables
- Transferability analysis
- Statistical analysis
- Reproducible experimental artifacts