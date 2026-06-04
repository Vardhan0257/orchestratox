# orchestratox_main.py
# Full OrchestraTox experiment — generates scenarios, evolves payloads,
# tests across 3 frameworks, produces paper-quality results table

import json
import time
import numpy as np
from pathlib import Path
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from scenario_generator import generate_scenario_bank
from payload_evolver import PayloadEvolver
from frameworks import CustomPipeline, LangGraphPipeline, AutoGenPipeline

# ── Configuration ─────────────────────────────────────────────────────────
N_SCENARIOS = 10          # number of auto-generated scenarios
N_TRIALS_PER_DEPTH = 5   # trials per (scenario, depth, framework) combination
DEPTHS_TO_TEST = [0, 2, 3]
EVOLVE_PAYLOADS = False     # set False to skip evolution and use base payloads
EVOLUTION_GENERATIONS = 3
EVOLUTION_POPULATION = 3
RESULTS_FILE = "orchestratox_final_results.json"

def run_scenario_on_framework(scenario, framework, depths, n_trials):
    """Run all depth conditions for one scenario on one framework."""
    results = []
    for depth in depths:
        def run_one_trial(trial_idx):
            try:
                temp = trial_idx * (0.9 / max(n_trials - 1, 1))
                return framework.run_trial(scenario, depth, temperature=temp)
            except Exception as e:
                return {
                    "framework": framework.name,
                    "attack_succeeded": False,
                    "injection_depth": depth,
                    "error": str(e),
                    "orchestrator_response": ""
                }

        with ThreadPoolExecutor(max_workers=3) as executor:
            depth_results = list(executor.map(run_one_trial, range(n_trials)))
        
        successes = sum(1 for r in depth_results if r["attack_succeeded"])
        asr = successes / len(depth_results) if depth_results else 0
        
        # Wilson confidence interval (better than normal approx for small N)
        ci_low, ci_high = wilson_ci(successes, len(depth_results))
        
        results.append({
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "domain": scenario.get("domain", "unknown"),
            "framework": framework.name,
            "depth": depth,
            "trials": len(depth_results),
            "successes": successes,
            "asr": asr,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
        
        print(f"    Depth {depth}: {successes}/{len(depth_results)} = {asr:.1%} "
              f"[{ci_low:.1%}, {ci_high:.1%}]")
    
    return results


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score confidence interval — correct for small samples."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denominator
    return max(0, center - margin), min(1, center + margin)


def compute_summary_table(all_results: list) -> dict:
    """Aggregate results across all scenarios into summary statistics."""
    summary = {}
    
    frameworks = list(set(r["framework"] for r in all_results))
    depths = sorted(set(r["depth"] for r in all_results))
    
    for framework in frameworks:
        summary[framework] = {}
        for depth in depths:
            depth_data = [r for r in all_results 
                         if r["framework"] == framework and r["depth"] == depth]
            
            if not depth_data:
                continue
            
            total_trials = sum(r["trials"] for r in depth_data)
            total_successes = sum(r["successes"] for r in depth_data)
            asr = total_successes / total_trials if total_trials > 0 else 0
            ci_low, ci_high = wilson_ci(total_successes, total_trials)
            
            # Per-scenario ASR variance (shows consistency)
            per_scenario_asrs = [r["asr"] for r in depth_data]
            asr_std = np.std(per_scenario_asrs) if len(per_scenario_asrs) > 1 else 0
            
            summary[framework][f"depth_{depth}"] = {
                "asr": asr,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "asr_std": asr_std,
                "total_trials": total_trials,
                "total_successes": total_successes,
                "n_scenarios": len(depth_data)
            }
    
    return summary


def print_results_table(summary: dict):
    """Print paper-quality results table."""
    print("\n" + "="*80)
    print("ORCHESTRATOX RESULTS — ATTACK SUCCESS RATE BY FRAMEWORK AND DEPTH")
    print("="*80)
    
    frameworks = list(summary.keys())
    depths = [0, 1, 2, 3]
    depth_labels = ["CLEAN", "DEPTH-1", "DEPTH-2", "DEPTH-3"]
    
    # Header
    header = f"{'Depth':<12}"
    for fw in frameworks:
        header += f" {fw:<25}"
    print(header)
    print("-" * (12 + 25 * len(frameworks)))
    
    for depth, label in zip(depths, depth_labels):
        row = f"{label:<12}"
        for fw in frameworks:
            key = f"depth_{depth}"
            if fw in summary and key in summary[fw]:
                d = summary[fw][key]
                row += f" {d['asr']:.1%} [{d['ci_low']:.1%},{d['ci_high']:.1%}]{'':<5}"
            else:
                row += f" {'N/A':<25}"
        print(row)
    
    print("="*80)
    
    # Cross-framework ASR lift
    print("\nATTACK LIFT (Injected - Clean ASR):")
    for fw in frameworks:
        if fw not in summary:
            continue
        clean = summary[fw].get("depth_0", {}).get("asr", 0)
        depth3 = summary[fw].get("depth_3", {}).get("asr", 0)
        lift = depth3 - clean
        print(f"  {fw}: {lift:.1%} lift at depth-3")
    
    print("="*80)


def main():
    print("="*60)
    print("ORCHESTRATOX: FULL EXPERIMENT")
    print("="*60)
    
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Generate scenario bank
    print("\n[1/4] Generating scenario bank...")
    scenarios = generate_scenario_bank(
        n_domains=N_SCENARIOS,
        output_file="results/scenario_bank.json"
    )
    print(f"Using {len(scenarios)} scenarios")
    
    # Step 2: Initialize frameworks
    print("\n[2/4] Initializing agent frameworks...")
    frameworks = [
        CustomPipeline(),
        AutoGenPipeline(),
        LangGraphPipeline()
    ]
    available_frameworks = [f for f in frameworks 
                           if not hasattr(f, 'available') or f.available]
    print(f"Available frameworks: {[f.name for f in available_frameworks]}")
    
    # Step 3: Optionally evolve payloads
    if EVOLVE_PAYLOADS:
        print("\n[3/4] Evolving payloads (adaptive attack)...")
        evolved_scenarios = []
        
        for i, scenario in enumerate(scenarios[:5]):  # evolve first 5 as demo
            print(f"\n  Scenario {i+1}: {scenario['domain']}")
            
            def trial_fn(payload):
                """Quick single trial to evaluate payload."""
                test_scenario = dict(scenario)
                test_scenario["injection_payload"] = payload
                fw = CustomPipeline()
                result = fw.run_trial(test_scenario, injection_depth=2)
                return result["attack_succeeded"]
            
            evolver = PayloadEvolver(
                target_action=scenario["target_action"],
                domain=scenario["domain"]
            )
            
            evolution_result = evolver.evolve(
                trial_function=trial_fn,
                generations=EVOLUTION_GENERATIONS,
                population_size=EVOLUTION_POPULATION,
                trials_per_payload=3
            )
            
            if evolution_result["best_asr"] > 0:
                scenario["injection_payload"] = evolution_result["best_payload"]
                scenario["evolved_asr"] = evolution_result["best_asr"]
                print(f"  Best evolved payload ASR: {evolution_result['best_asr']:.1%}")
            
            evolved_scenarios.append(scenario)
        
        # Keep remaining scenarios with original payloads
        evolved_scenarios.extend(scenarios[5:])
        scenarios = evolved_scenarios
        
        # Save evolved scenarios
        with open("results/evolved_scenarios.json", "w") as f:
            json.dump(scenarios, f, indent=2)
    else:
        print("\n[3/4] Skipping payload evolution (EVOLVE_PAYLOADS=False)")
    
    # Step 4: Run full experiment
    print(f"\n[4/4] Running experiments...")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Frameworks: {[f.name for f in available_frameworks]}")
    print(f"  Depths: {DEPTHS_TO_TEST}")
    print(f"  Trials per condition: {N_TRIALS_PER_DEPTH}")
    total_calls = (len(scenarios) * len(available_frameworks) * 
                   len(DEPTHS_TO_TEST) * N_TRIALS_PER_DEPTH * 3)
    print(f"  Estimated LLM calls: {total_calls}")
    print(f"  Estimated time: {total_calls * 5 / 60:.0f} minutes")
    
    all_results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}/{len(scenarios)}: {scenario['domain']}")
        
        for framework in available_frameworks:
            print(f"  Framework: {framework.name}")
            results = run_scenario_on_framework(
                scenario, framework, DEPTHS_TO_TEST, N_TRIALS_PER_DEPTH
            )
            all_results.extend(results)
        
        # Save intermediate results
        with open("results/intermediate_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Compute and print summary
    summary = compute_summary_table(all_results)
    print_results_table(summary)
    
    # Save final results
    output = {
        "config": {
            "n_scenarios": len(scenarios),
            "n_trials_per_depth": N_TRIALS_PER_DEPTH,
            "depths": DEPTHS_TO_TEST,
            "frameworks": [f.name for f in available_frameworks],
            "evolved_payloads": EVOLVE_PAYLOADS
        },
        "summary": summary,
        "all_results": all_results,
        "scenarios_used": len(scenarios)
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFull results saved to {RESULTS_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()