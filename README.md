# OrchestraTox

**Evaluating Sanitization Coverage Asymmetry and Production-Proxy Divergence in Multi-Agent LLM Systems**


## Overview

This repository contains the code, scenario bank, and experimental results for our study of trust propagation attacks in multi-agent LLM systems. We demonstrate that:

1. Production AutoGen achieves **54.6 percentage points of attack lift** — 4.8× higher than a simplified proxy of the same architecture.
2. LLM-based input sanitization reduces direct injection to **3.8% ASR** but fails against trust propagation attacks through sub-agent channels (**23.8% ASR**, Fisher's exact p < 0.001).
3. Attack success concentrates in **high-ambiguity delegated decision domains** consistently across LLaMA-3.2-3B, Mistral-7B, GPT-3.5-Turbo, and GPT-4.1-mini.

## System Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- 8GB VRAM minimum (consumer GPU sufficient — tested on RTX 4060)
- OpenAI API key (optional — only for GPT validation experiments, total cost under $1.00)

## Installation

```bash
# Clone repository
git clone https://github.com/YOURUSERNAME/orchestratox.git
cd orchestratox

# Install Python dependencies
pip install -r requirements.txt

# Pull required models via Ollama
ollama pull llama3.2
ollama pull mistral:7b-instruct-q4_K_M
ollama pull nomic-embed-text
```

## Quick Start — Reproduce Main Results

```bash
# Start Ollama (keep running in separate terminal)
ollama serve

# Run main experiment (all 4 frameworks, 13 scenarios, 4 depths, 20 trials)
# Estimated time: 8-12 hours on consumer GPU
python orchestratox_main.py

# Validate results with independent Mistral judge
python mistral_judge_validator.py

# Generate analysis tables
python analyze_results.py
```

## Reproducing Individual Experiments

| Experiment | Script | Runtime | Cost |
|---|---|---|---|
| Main ASR evaluation | `orchestratox_main.py` | ~10 hours | Free |
| Defense boundary | `production_gap_experiment.py` | ~3 hours | Free |
| Conversation history ablation | `autogen_ablation.py` | ~5 hours | Free |
| Adaptive attacker | `adaptive_attack.py` | ~2 hours | Free |
| Defense evaluation | `defense_experiment.py` | ~2 hours | Free |
| Memory persistence | `memory_attack.py` | ~1 hour | Free |
| Judge validation (Mistral) | `mistral_judge_validator.py` | ~1 hour | Free |
| Cross-architecture (Mistral orch.) | `mistral_orchestrator_experiment.py` | ~4 hours | Free |
| GPT-3.5-Turbo validation | `gpt35_validation.py` | ~45 min | ~$0.29 |
| GPT-4.1-mini validation | `gpt4mini_validation.py` | ~60 min | ~$0.37 |
| Real poisoning demo | `real_poisoning_demo.py` | ~10 min | Free |
| Annotation agreement | `compute_annotation_agreement.py` | ~5 min | Free |
| Ambiguity regression | `ambiguity_regression.py` | ~5 min | Free |

## For GPT Validation Experiments

Set your OpenAI API key in the script files (`gpt35_validation.py`, `gpt4mini_validation.py`):

```python
API_KEY = "sk-your-key-here"
```

Total cost for all API experiments: under $1.00.

## Repository Structure

```
orchestratox/
├── orchestratox_main.py          # Main experiment runner
├── frameworks.py                 # CustomPipeline, AutoGen-style, LangGraph
├── autogen_pipeline.py           # Production AutoGen (v0.2.35) pipeline
├── scenario_generator.py         # LLM-based scenario generation
├── judge_validator.py            # String-match judge
├── mistral_judge_validator.py    # Mistral independent judge + Cohen's kappa
├── defense_experiment.py         # Keyword + LLM sanitization defense
├── production_gap_experiment.py  # Defense boundary experiment
├── adaptive_attack.py            # Filter-aware adaptive attacker
├── autogen_ablation.py           # Conversation history ablation
├── mistral_orchestrator_experiment.py  # Cross-architecture validation
├── memory_attack.py              # Cross-session memory persistence
├── real_poisoning_demo.py        # End-to-end document retrieval demo
├── gpt35_validation.py           # GPT-3.5-Turbo API validation
├── gpt4mini_validation.py        # GPT-4.1-mini API validation
├── compute_annotation_agreement.py  # Inter-annotator Cohen's kappa
├── ambiguity_regression.py       # Spearman correlation analysis
├── analyze_results.py            # Result analysis and tables
├── requirements.txt              # Python dependencies
├── results/                      # Pre-computed experimental results
│   ├── scenario_bank.json        # 13 validated attack scenarios
│   ├── orchestratox_final_results.json   # Main experiment results
│   ├── judge_validated_results.json      # Mistral-validated results
│   ├── production_gap_results.json       # Defense boundary results
│   ├── autogen_ablation.json             # History ablation results
│   ├── adaptive_attack_results.json      # Adaptive attacker results
│   ├── defense_results.json              # Defense evaluation results
│   ├── mistral_orchestrator_results.json # Cross-architecture results
│   ├── gpt35_validation.json             # GPT-3.5-Turbo results
│   ├── gpt4mini_results.json             # GPT-4.1-mini results
│   ├── memory_results.json               # Memory persistence results
│   ├── annotation_agreement.json         # Annotator agreement
│   └── ambiguity_regression.json         # Regression results
└── README.md
```

## Pre-Computed Results

All experimental results are provided in `results/` as JSON files. You can reproduce any table or figure in the paper by running the corresponding analysis scripts on these pre-computed results without re-running the full experiments.

```bash
# Reproduce all tables from pre-computed results
python analyze_results.py

# Reproduce annotation agreement table
python compute_annotation_agreement.py

# Reproduce ambiguity regression
python ambiguity_regression.py
```

## Scenario Bank

The 13 attack scenarios used in evaluation are in `results/scenario_bank.json`. Each scenario contains:

- `domain`: Operational domain (healthcare, insurance, etc.)
- `task`: The orchestrator's assigned task
- `query`: What sub-agents research
- `injection_payload`: The adversarial payload (authority-framed, ambiguous)
- `target_action`: The dangerous action the attack causes
- `benign_action`: The correct safe action
- `available_actions`: Full action set available to orchestrator

## Ethical Considerations

This research was conducted for defensive purposes. All experiments used locally hosted open-source models in simulated pipelines. No production systems were attacked. Findings were disclosed to AutoGen and LangGraph maintainers prior to publication. The attack techniques described are applicable only by parties with write access to data sources retrieved by sub-agents (web pages, support tickets, code repositories) — capabilities already available to any web author or code contributor.

## Hardware Used

All primary experiments were conducted on a consumer laptop with:
- NVIDIA RTX 4060 (8GB VRAM)
- 16GB RAM
- No cloud compute or paid GPU services

## Citation

```bibtex
@inproceedings{orchestratox2026,
  title={Evaluating Sanitization Coverage Asymmetry and Production-Proxy 
         Divergence in Multi-Agent LLM Systems},
  author={[Author Name]},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2026}
}
```

## License

MIT License. See LICENSE file for details.