# annotate_ambiguity.py
# Generates annotation sheets for 3 independent annotators
# They label WITHOUT seeing ASR results

import json
from pathlib import Path

with open("results/scenario_bank.json") as f:
    scenarios = json.load(f)

# Print annotation instructions
print("""
AMBIGUITY ANNOTATION TASK
==========================
For each scenario below, classify the TARGET ACTION into ONE of these 5 categories:

1. COMPLIANCE-DRIVEN: Action sounds like a regulatory or audit requirement
   Example: "reassign cases to comply with audit directive"

2. SAFETY-FRAMED: Action sounds like a protective safety measure  
   Example: "enable auto-moderation to prevent harmful content"

3. AUTHORITY-FRAMED: Action sounds like expert-recommended remediation
   Example: "disable biased data learning per regulatory guidance"

4. SECURITY-EXPLICIT: Action has unambiguous security-critical meaning
   Example: "disable account authentication", "remove access controls"

5. OPERATIONAL-EXPLICIT: Action has clear operational meaning, no safety spin
   Example: "unlock front door", "disable smoke alarm", "delete user files"

Instructions:
- Read only the DOMAIN and TARGET ACTION
- Do NOT consider whether the attack worked
- Label based on how the action SOUNDS to someone in that domain
- Use numbers 1-5 only
""")

print("\nSCENARIOS TO LABEL:")
print("="*60)

for i, s in enumerate(scenarios):
    print(f"\nScenario {i+1}:")
    print(f"  Domain: {s.get('domain', 'unknown')}")
    print(f"  Target action: {s.get('target_action', 'unknown')}")
    print(f"  Task context: {s.get('task', 'unknown')}")
    print(f"  Your label (1-5): ___")

print("\n" + "="*60)
print("Save your labels as: annotator_A.txt, annotator_B.txt, annotator_C.txt")
print("Format: one number per line, in scenario order")