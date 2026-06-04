# diagnose_realautogen.py
from autogen_pipeline import RealAutoGenPipeline
import json

fw = RealAutoGenPipeline()
print("Available:", fw.available)

with open("results/scenario_bank.json") as f:
    scenarios = json.load(f)

scenario = next(s for s in scenarios if "healthcare" in s.get("domain", ""))
print("Testing scenario:", scenario["domain"])
print("Target action:", scenario["target_action"])

result = fw.run_trial(scenario, injection_depth=3, temperature=0.3)
print("Attack succeeded:", result["attack_succeeded"])
print("Action taken:", result["action_taken"])
print("Response (first 300 chars):")
print(result.get("orchestrator_response", "EMPTY")[:300])