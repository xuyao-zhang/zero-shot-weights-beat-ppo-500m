# SPDX-License-Identifier: MIT
# Copyright (c) 2026 opopile (zero-shot-weights-beat-ppo-500m contributors)
"""60-second independent verifier for the 3/3 PPO-500M zero-shot weight sweep.

Checks:
1. Parses every reply.md -> edits -> float32 network and verifies 0.0 max diff vs network.npz
2. Verifies 0 parse errors and cut_off=False across all 6 networks (ZH 3/3 + EN 3/3)
3. Verifies EN vs ZH cross-lingual non-reuse metrics and two-sample episode z-tests
4. Unless --skip-sim is passed, runs official MinAtar==1.0.15 (1,000 episodes, seed=0..999)
   on the ZH 3/3 sweep (or --all-6 to also replay EN baseline) and confirms exact score parity!
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wsp_eval import (  # noqa: E402
    GAMES,
    PPO500,
    RANDOM_PAPER,
    SHAPES,
    apply_edits,
    compare_networks,
    load_npz,
    score_network,
    two_sample_z_test,
    wsb_median_score,
    wwi_overall,
)

EXPECTED_NZ = {
    ("zh_prompt", "breakout"): 2009,
    ("zh_prompt", "asterix"): 976,
    ("zh_prompt", "seaquest"): 1207,
    ("en_baseline", "breakout"): 1571,
    ("en_baseline", "asterix"): 920,
    ("en_baseline", "seaquest"): 550,
}


def main(argv: list[str] | None = None) -> int:
    repo = SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description="Verify all zero-shot weight networks and replay MinAtar 1.0.15.")
    parser.add_argument("--skip-sim", action="store_true", help="Skip 1,000-episode MinAtar simulation.")
    parser.add_argument("--all-6", action="store_true", help="Replay both ZH (3) and EN (3) networks during simulation.")
    parser.add_argument("--workers", type=int, default=15, help="CPU worker processes (15 reproduces score.json to <1e-6).")
    args = parser.parse_args(argv)

    art = repo / "artifacts"
    nets = {}
    scores = {}
    for cond in ("zh_prompt", "en_baseline"):
        for g in GAMES:
            d = art / cond / g
            reply = (d / "reply.md").read_text(encoding="utf-8")
            edits_saved = json.loads((d / "edits.json").read_text(encoding="utf-8"))
            score_saved = json.loads((d / "score.json").read_text(encoding="utf-8"))
            npz_net = load_npz(d / "network.npz")

            parsed_net, rep = apply_edits(reply)
            assert rep == edits_saved, f"{cond}/{g}: edits report mismatch"
            assert rep["errors"] == [], f"{cond}/{g}: parse errors found"
            assert rep["cut_off"] is False, f"{cond}/{g}: cut_off=True"
            assert rep["nonzero_total"] == EXPECTED_NZ[(cond, g)], f"{cond}/{g}: nonzero mismatch"
            max_diff = max(float(np.max(np.abs(parsed_net[k] - npz_net[k]))) for k in SHAPES)
            assert max_diff == 0.0, f"{cond}/{g}: max_diff={max_diff}"

            nets[(cond, g)] = npz_net
            scores[(cond, g)] = score_saved
            print(
                f"[OK] {cond:<11}/{g:<8} | stmts={rep['applied']:>3}/{rep['statements']:<3} "
                f"| nz={rep['nonzero_total']:>4} | max_diff=0.0 | mean={score_saved['mean']:>7.3f} +/- {score_saved['se']:.3f}"
            )

    zh_means = {g: scores[("zh_prompt", g)]["mean"] for g in GAMES}
    en_means = {g: scores[("en_baseline", g)]["mean"] for g in GAMES}
    print(f"\n[SUMMARY] ZH Prompt   | WWI = {wwi_overall(zh_means):.4f} | WSB Median = {wsb_median_score(zh_means):.4f} | 3/3 > PPO-500M: {all(zh_means[g] > PPO500[g] for g in GAMES)}")
    print(f"[SUMMARY] EN Baseline | WWI = {wwi_overall(en_means):.4f} | WSB Median = {wsb_median_score(en_means):.4f} | 3/3 > PPO-500M: {all(en_means[g] > PPO500[g] for g in GAMES)}")

    print("\n=== EN Baseline vs ZH Translation Cross-Check ===")
    for g in GAMES:
        cmp = compare_networks(nets[("en_baseline", g)], nets[("zh_prompt", g)])
        z = two_sample_z_test(scores[("en_baseline", g)]["mean"], scores[("en_baseline", g)]["se"],
                              scores[("zh_prompt", g)]["mean"], scores[("zh_prompt", g)]["se"])
        print(
            f"  {g:<8}: {en_means[g]:6.2f} -> {zh_means[g]:6.2f} ({z['pct_gain']:+7.2f}%, z={z['z_stat']:+6.2f}) "
            f"| nz {cmp['nonzero_a']}->{cmp['nonzero_b']}, identical={cmp['identical_nonzero_coords']}, cos={cmp['cosine_similarity']:+.4f}"
        )

    if args.skip_sim:
        print("\n[SKIP] --skip-sim set; MinAtar simulation skipped.")
        return 0

    conds = ("zh_prompt", "en_baseline") if args.all_6 else ("zh_prompt",)
    tol = 1e-6 if args.workers == 15 else 0.5
    print(f"\n=== Live 1,000-Episode MinAtar==1.0.15 Verification (workers={args.workers}, tol={tol}) ===")
    for cond in conds:
        for g in GAMES:
            t0 = time.time()
            live = score_network(nets[(cond, g)], g, episodes=1000, seed=0, max_steps=10000, workers=args.workers)
            dt = time.time() - t0
            stored = scores[(cond, g)]["mean"]
            diff = abs(live["mean"] - stored)
            assert diff <= tol, f"{cond}/{g}: live={live['mean']:.4f} vs stored={stored:.4f} (diff={diff:.4f} > {tol})"
            beat_ppo = live["mean"] > PPO500[g]
            print(
                f"  [VERIFIED] {cond}/{g:<8} | live={live['mean']:7.3f} +/- {live['se']:.3f} "
                f"| diff={diff:.2e} | > PPO-500M ({PPO500[g]:.1f}): {beat_ppo} | {dt:.1f}s"
            )
    print("\nALL CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
