# OrchestraTox – Impact 2 Experimental Protocol

## Objective

Evaluate whether prompt-injection attacks discovered on one LLM transfer to other LLMs under an identical multi-agent orchestration framework.

---

## Benchmark

Use the frozen human-reviewed benchmark from Impact 1.

Scenario file:

results/scenario_bank_human_reviewed.json

Do not modify the benchmark.

---

## Experimental Design

The orchestration framework, prompts, temperatures, scenarios, action space, and evaluation logic remain identical to Impact 1.

Only the underlying LLM changes.

---

## Independent Variable

LLM

---

## Controlled Variables

- Human-reviewed benchmark
- Prompt templates
- Injection depths
- Available actions
- Evaluation logic
- Trial count
- Temperature
- Randomisation procedure

---

## Dependent Variable

Target-action selection.

---

## Primary Outcome

Attack success rate for each model.

---

## Secondary Outcomes

- Cross-model agreement
- Cross-model disagreement
- Scenario-specific transferability
- Overall transferability score

---

## Statistical Analysis

Compute:

- Per-model attack success
- Pairwise model comparisons
- Transferability matrix
- Bootstrap confidence intervals
- Aggregate comparison tables

---

## Reproducibility

Every experiment must record:

- model name
- benchmark SHA-256
- experiment timestamp
- token usage
- API cost
- trial count
- protocol version