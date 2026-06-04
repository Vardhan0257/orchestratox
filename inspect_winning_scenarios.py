# inspect_winning_scenarios.py
import json

with open("results/scenario_bank.json") as f:
    scenarios = json.load(f)

winning_ids = ["4286c0f7", "a12197c7", "f862a24f", "f39d72ac", "be6e8a53"]

for s in scenarios:
    if s.get("scenario_id") in winning_ids:
        print(f"\n{'='*60}")
        print(f"Domain: {s['domain']}")
        print(f"Task: {s['task']}")
        print(f"Target action: {s['target_action']}")
        print(f"Benign action: {s.get('benign_action', 'N/A')}")
        print(f"Injection payload: {s['injection_payload'][:200]}")
        print(f"Available actions: {s['available_actions']}")