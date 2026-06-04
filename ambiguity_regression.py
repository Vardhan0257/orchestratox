# ambiguity_regression.py — fixed to handle aggregated results format

import json
import numpy as np
from scipy import stats

with open("results/orchestratox_final_results.json") as f:
    results_data = json.load(f)

AMBIGUITY_SCORES = {
    "healthcare AI diagnosis system": 4,
    "content moderation platform": 4,
    "insurance claim processing": 5,
    "supply chain management": 3,
    "legal document processing": 3,
    "e-commerce recommendation": 3,
    "smart home automation": 1,
    "autonomous code deployment": 2,
    "cybersecurity incident response": 1,
    "HR candidate screening": 2,
    "customer data management": 2,
    "academic research assistant": 2,
    "government permit approval": 3,
}

all_results = results_data.get("all_results", [])
print(f"Aggregated records: {len(all_results)}")
print(f"Sample: {all_results[0]}")

# Reconstruct trial-level data from aggregated successes/trials
ambiguity_values = []
success_values = []
domain_list = []

for record in all_results:
    domain = record.get("domain", "")
    depth = record.get("depth", 0)
    successes = record.get("successes", 0)
    trials = record.get("trials", 0)
    
    if depth == 0:  # skip clean baseline
        continue
    if not domain or trials == 0:
        continue
    
    ambig = AMBIGUITY_SCORES.get(domain, 3)
    
    # Expand: successes times success=1, (trials-successes) times success=0
    for _ in range(int(successes)):
        ambiguity_values.append(ambig)
        success_values.append(1)
        domain_list.append(domain)
    for _ in range(int(trials) - int(successes)):
        ambiguity_values.append(ambig)
        success_values.append(0)
        domain_list.append(domain)

X = np.array(ambiguity_values)
y = np.array(success_values)

print(f"\nExpanded trial-level data: {len(y)} observations")
print(f"Overall injected ASR: {y.mean():.1%}")

print(f"\nASR by ambiguity score:")
for score in sorted(set(X)):
    mask = X == score
    asr = y[mask].mean()
    n = int(mask.sum())
    s = int(y[mask].sum())
    print(f"  Score {score}: {asr:.1%} ({s}/{n})")

# Per-domain ASR for Spearman
domain_asrs = {}
domain_ambig = {}
for domain, ambig, success in zip(domain_list, ambiguity_values, success_values):
    if domain not in domain_asrs:
        domain_asrs[domain] = []
        domain_ambig[domain] = ambig
    domain_asrs[domain].append(success)

domains = list(domain_asrs.keys())
per_domain_asr = [np.mean(domain_asrs[d]) for d in domains]
per_domain_ambig = [domain_ambig[d] for d in domains]

print(f"\nPer-domain results:")
for d, asr, ambig in sorted(zip(domains, per_domain_asr, per_domain_ambig),
                              key=lambda x: x[2], reverse=True):
    print(f"  {d[:35]:<35} ambig={ambig} asr={asr:.1%}")

# Spearman correlation
rho, p_spearman = stats.spearmanr(per_domain_ambig, per_domain_asr)
print(f"\nSpearman ρ = {rho:.3f}, p = {p_spearman:.4f}")

# Point-biserial
r, p_pb = stats.pointbiserialr(X, y)
print(f"Point-biserial r = {r:.3f}, p = {p_pb:.4f}")

if p_spearman < 0.05 and rho > 0.4:
    verdict = "STRONG"
    paper_text = (f"Ambiguity score significantly predicts domain-level ASR "
                 f"(Spearman ρ = {rho:.2f}, p = {p_spearman:.3f}).")
elif p_spearman < 0.10 and rho > 0.3:
    verdict = "MODERATE"
    paper_text = (f"Ambiguity score shows positive correlation with domain-level ASR "
                 f"(Spearman ρ = {rho:.2f}, p = {p_spearman:.3f}).")
else:
    verdict = "NOT SIGNIFICANT"
    paper_text = "Report cross-scale consistency as primary evidence instead."

print(f"\nVerdict: {verdict}")
print(f"Paper text: {paper_text}")

with open("results/ambiguity_regression.json", "w") as f:
    json.dump({
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "point_biserial_r": float(r),
        "point_biserial_p": float(p_pb),
        "verdict": verdict,
        "paper_text": paper_text,
        "n_expanded_trials": len(y),
        "per_domain": {d: {"asr": float(np.mean(domain_asrs[d])),
                           "ambiguity": domain_ambig[d]} for d in domains}
    }, f, indent=2)
print("Saved.")