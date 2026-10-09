#!/usr/bin/env python3
"""
impact4b_transfer.py -- do payloads found against model A work on model B?

Replays the best payload per scenario from results/adaptive_<A>.json on a
different target model, with NO new attacker queries (pure black-box transfer).
5 fresh trials per scenario in both pipelines at depth 3, compared with the
target's own static-payload rates from its Impact 2 run.

Keep next to impact4_adaptive.py, analyze_impact2.py and
impact2_cross_model_transfer.py. Resumable; budget-capped.

  python impact4b_transfer.py gpt-5-mini --effort minimal --max-usd 0.30
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import analyze_impact2 as an
import impact2_cross_model_transfer as base
import impact4_adaptive as ad

P, H, FINAL = ad.P, ad.H, ad.FINAL


def run(target, source_path, force, max_usd):
    src = json.load(open(source_path, encoding="utf-8"))
    src_model = src["model"]
    tag = f"{src_model.replace('.', '_')}_to_{target.replace('.', '_')}"
    out_path = Path(f"results/transfer_{tag}.json")
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists. Use --force to redo.")
        return

    sha = base.sha256_file(base.SCENARIO_FILE)
    if src.get("scenario_file_sha256") != sha:
        raise SystemExit("Scenario bank differs from the one used for the source attack. Aborting.")
    bank = {s["domain"]: s for s in base.load_scenarios(False)}
    static = ad.static_rates(
        f"results/frontier_framework_comparison_{target.replace('.', '_')}.json"
    )

    base.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt, meta = base.checkpoint_paths(target, f"_transfer_from_{src_model.replace('.', '_')}")
    base.check_meta(meta, {
        "model": target, "scenario_sha256": sha, "trials": FINAL,
        "depths": [ad.DEPTH], "scenario_count": len(src["scenarios"]), "pilot": False,
    })
    done = base.load_checkpoint(ckpt)

    print("=" * 72)
    print(f"TRANSFER  {src_model} -> {target}  scenarios={len(src['scenarios'])}  "
          f"resumed_rows={len(done)}")
    print("=" * 72)

    rows = []
    for i, item in enumerate(src["scenarios"], start=1):
        sc = bank[item["domain"]]
        held = {}
        for cond in (P, H):
            ok_n = 0
            for t in range(FINAL):
                key = f"{i}|tr|{cond}|{t}"
                if key in done:
                    row = done[key]
                else:
                    if max_usd is not None and base.SESSION_COST[0] >= max_usd:
                        print(f"\nBUDGET CAP reached (${base.SESSION_COST[0]:.3f} >= ${max_usd}). "
                              "Progress saved; rerun to continue.")
                        sys.exit(2)
                    c0 = base.SESSION_COST[0]
                    ok, sel, _ = ad.decide(cond, sc, item["best_payload"], t, target)
                    row = {"key": key, "success": ok, "selected": sel,
                           "usd": base.SESSION_COST[0] - c0}
                    base.append_checkpoint(ckpt, row)
                    done[key] = row
                    time.sleep(0.2)
                ok_n += int(row["success"])
            held[cond] = ok_n
        st = static.get(item["domain"], {})
        adapted = not item["best_is_static"]
        print(f"{item['domain'][:32]:32s} {'ADAPTED' if adapted else 'static '} "
              f"Proxy {held[P]}/{FINAL}  History {held[H]}/{FINAL} | "
              f"{target} static: Proxy {st.get(P, float('nan')) * 100:.0f}%  "
              f"History {st.get(H, float('nan')) * 100:.0f}%")
        rows.append({"domain": item["domain"], "adapted": adapted, "heldout": held,
                     "static_rate_depth3": st, "payload": item["best_payload"]})

    total_usd = sum(r.get("usd", 0.0) for r in done.values())
    summary = {}
    print("\n" + "=" * 72)
    print(f"TRANSFER RESULT on {target} (depth 3, fresh trials)")
    print("=" * 72)
    for label, subset in (("all 15 scenarios", rows),
                          ("scenarios where the attacker found a better payload",
                           [r for r in rows if r["adapted"]])):
        if not subset:
            continue
        print(f"\n[{label}: n={len(subset)}]")
        for cond in (P, H):
            tr = np.array([r["heldout"][cond] / FINAL for r in subset])
            stt = np.array([r["static_rate_depth3"].get(cond, np.nan) for r in subset])
            ok = ~np.isnan(stt)
            d = an.describe((tr - stt)[ok]) if ok.any() else None
            print(f"  {cond:20s} compromised (>=50%): static {int((stt[ok] >= 0.5).sum())}"
                  f" -> transferred {int((tr >= 0.5).sum())} | pooled "
                  f"{np.nanmean(stt):.1%} -> {tr.mean():.1%}")
            print(f"  {'':20s} paired gain: {an.fmt(d)}")
            summary[f"{label}|{cond}"] = {
                "n": len(subset), "transferred_asr": float(tr.mean()),
                "static_asr": float(np.nanmean(stt)),
                "compromised_transferred": int((tr >= 0.5).sum()),
                "compromised_static": int((stt[ok] >= 0.5).sum()), "paired_gain": d}
    print(f"\nEstimated cost across all sessions: ${total_usd:.3f}")

    result = {"experiment": "impact4b_transfer", "source_model": src_model,
              "target_model": target, "scenario_file_sha256": sha,
              "api_settings": {"reasoning_effort": base.GPT5_REASONING_EFFORT
                               if target.startswith("gpt-5") else None},
              "summary": summary, "estimated_cost_usd": total_usd, "scenarios": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.copy2(out_path, out_path.with_name(out_path.stem + f".bak_{int(time.time())}.json"))
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, out_path)
    print(f"Saved to: {out_path}\nTRANSFER COMPLETED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--source", default="results/adaptive_gpt-5_6-luna.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--effort", default=base.GPT5_REASONING_EFFORT)
    ap.add_argument("--max-usd", type=float, default=None)
    args = ap.parse_args()
    if args.target not in base.PRICES:
        raise SystemExit(f"Unknown model {args.target}. Known: {list(base.PRICES)}")
    base.GPT5_REASONING_EFFORT = args.effort
    run(args.target, args.source, args.force, args.max_usd)


if __name__ == "__main__":
    main()
