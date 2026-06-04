# logistic_regression.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats

# Load individual trial results
with open("results/individual_trials.json") as f:
    trials = json.load(f)

# Ambiguity scores from your validation experiment
with open("results/vulnerability_condition_validation.json") as f:
    vuln_data = json.load(f)

ambiguity_map = {
    d: v["ambiguity"] 
    for d, v in vuln_data.get("per_scenario", {}).items()
}

framework_map = {
    "CustomPipeline": 0, "AutoGen": 1, 
    "RealAutoGen": 2, "LangGraph": 3
}

rows = []
for trial in trials:
    domain = trial.get("domain", "")
    framework = trial.get("framework", "")
    depth = trial.get("depth", trial.get("injection_depth", 0))
    success = int(trial.get("attack_succeeded", False))
    ambiguity = ambiguity_map.get(domain, 3.0)
    fw_code = framework_map.get(framework, 0)
    
    rows.append({
        "success": success,
        "ambiguity": ambiguity,
        "depth": depth,
        "framework": fw_code,
        "is_production": int(framework == "RealAutoGen")
    })

df = pd.DataFrame(rows)
df = df[df["depth"] > 0]  # exclude clean baseline trials

X = df[["ambiguity", "depth", "is_production"]].values
y = df["success"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Compute odds ratios and confidence intervals
coefs = model.coef_[0]
odds_ratios = np.exp(coefs)

# Bootstrap CIs
n_boot = 1000
boot_ors = []
rng = np.random.RandomState(42)
for _ in range(n_boot):
    idx = rng.choice(len(X_scaled), len(X_scaled), replace=True)
    m = LogisticRegression(random_state=42)
    m.fit(X_scaled[idx], y[idx])
    boot_ors.append(np.exp(m.coef_[0]))

boot_ors = np.array(boot_ors)
ci_lo = np.percentile(boot_ors, 2.5, axis=0)
ci_hi = np.percentile(boot_ors, 97.5, axis=0)

features = ["Ambiguity score", "Injection depth", "Production framework"]
print("\nLogistic Regression: Predictors of Attack Success")
print("="*60)
print(f"{'Variable':<25} {'OR':>8} {'95% CI':>20}")
print("-"*60)
for i, feat in enumerate(features):
    print(f"{feat:<25} {odds_ratios[i]:>8.2f}  [{ci_lo[i]:.2f}, {ci_hi[i]:.2f}]")

# Save for paper
import json
results = {
    "variables": features,
    "odds_ratios": odds_ratios.tolist(),
    "ci_lo": ci_lo.tolist(),
    "ci_hi": ci_hi.tolist()
}
with open("results/logistic_regression.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/logistic_regression.json")