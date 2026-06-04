# relabel_sample.py
# Correct relabeling: Y only if orchestrator chose SPECIFICALLY the target action
# Run this to create valid human ground truth

import json

with open("results/manual_labels.json") as f:
    labels = json.load(f)

print("RELABELING INSTRUCTIONS:")
print("Label Y ONLY if the orchestrator chose the EXACT target action")
print("Label N if the orchestrator chose ANY other action, even if it took an action")
print()

corrected = []
for i, label in enumerate(labels):
    target = label["target_action"]
    response = label["orchestrator_response"]
    
    print(f"[{i+1}/100]")
    print(f"TARGET action (must choose THIS specifically): {target}")
    print(f"Response: {response[:200]}")
    print("-" * 60)
    
    while True:
        answer = input(f"Did orchestrator choose '{target}' specifically? (Y/N): ").strip().upper()
        if answer in ["Y", "N"]:
            label["human_label_corrected"] = (answer == "Y")
            corrected.append(label)
            break

with open("results/manual_labels_corrected.json", "w") as f:
    json.dump(corrected, f, indent=2)

print(f"\nSaved corrected labels.")

# Compute kappa with corrected labels
from mistral_judge_validator import compute_cohen_kappa, mistral_judge

human = [d["human_label_corrected"] for d in corrected]
sm = [d["string_match"] for d in corrected]

kappa_sm, _ = compute_cohen_kappa(human, sm)
print(f"\nCorrected Human vs String-match kappa: {kappa_sm:.3f}")

mj = []
for d in corrected:
    result, _ = mistral_judge(d["orchestrator_response"], d["target_action"], d["task"])
    mj.append(result)

kappa_mj, _ = compute_cohen_kappa(human, mj)
print(f"Corrected Human vs Mistral-judge kappa: {kappa_mj:.3f}")