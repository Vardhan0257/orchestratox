# compute_annotation_agreement.py
import numpy as np
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import json

annotators = {}
for name in ["A", "B", "C"]:
    try:
        with open(f"annotator_{name}.txt") as f:
            labels = [int(line.strip()) for line in f 
                      if line.strip().isdigit()]
            annotators[name] = labels
            print(f"Annotator {name}: {len(labels)} labels — {labels}")
    except FileNotFoundError:
        print(f"Missing annotator_{name}.txt")

if len(annotators) < 2:
    print("Need at least 2 annotator files.")
else:
    kappas = []
    for (a, la), (b, lb) in combinations(annotators.items(), 2):
        min_len = min(len(la), len(lb))
        k = cohen_kappa_score(la[:min_len], lb[:min_len])
        kappas.append(k)
        print(f"Cohen's κ ({a} vs {b}): {k:.3f}")

    mean_kappa = float(np.mean(kappas))
    print(f"\nMean inter-annotator κ: {mean_kappa:.3f}")

    if mean_kappa >= 0.6:
        verdict = "Substantial agreement. Taxonomy is reliable."
    elif mean_kappa >= 0.4:
        verdict = "Moderate agreement. Acceptable for publication."
    else:
        verdict = "Weak agreement. Taxonomy categories may need refinement."

    print(f"Verdict: {verdict}")

    with open("results/annotation_agreement.json", "w") as f:
        json.dump({
            "annotators": annotators,
            "pairwise_kappas": kappas,
            "mean_kappa": mean_kappa,
            "verdict": verdict
        }, f, indent=2)
    print("Saved to results/annotation_agreement.json")