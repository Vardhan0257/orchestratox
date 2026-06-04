# quick_judge_test.py
from judge_validator import judge_action
import json

with open("results/individual_trials.json") as f:
    trials = json.load(f)

# Test first 5 successes
successes = [t for t in trials if t.get("attack_succeeded")][:5]
for t in successes:
    response = t.get("orchestrator_response", "")
    target = t.get("target_action", "")
    task = t.get("task", "")
    print(f"\nTarget: {target}")
    print(f"Response: {response[:150]}")
    result = judge_action(response, target, task)
    print(f"Judge: {result}")