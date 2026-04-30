"""Head-to-head paired comparison between two TFLite candidates.

Runs both candidate models on the cleaned originals-only test set, then
computes:

- per-model accuracy with Wilson 95 % CI (capture-level)
- per-model macro-F1 with cluster-bootstrap 95 % CI (capture-level)
- exact two-sided McNemar p-value on the paired discordant-capture counts
- discordant-capture summary table

This is the primary statistical test in the lexicographic decision rule
(see docs/branch-audit.md and bench/results/calibration_report.md).
At the n=60 capture sample size, asymptotic chi-square McNemar is unsafe
so we use the exact binomial form.

Usage:

    python -m python.bench.compare_models \\
        --baseline python/gen/baseline_model.tflite \\
        --challenger python/gen/model.tflite

Outputs `bench/results/mcnemar_comparison.md`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from bench.eval_branches import evaluate as evaluate_tflite  # noqa: E402
from bench.stats import (  # noqa: E402
    cluster_bootstrap_macro_f1,
    exact_mcnemar,
    macro_f1,
    wilson_ci,
)

DEFAULT_RESULTS = ROOT / "bench" / "results"
DEFAULT_REPORT = DEFAULT_RESULTS / "mcnemar_comparison.md"
LABELS = ("Amine", "Rifki", "Jakub")


def label_name(idx: int) -> str:
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return str(idx)


def evaluate(model_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds, probs, _meta = evaluate_tflite(
        model_path=str(model_path),
        x=x,
        y=y,
        norm="pm1",
        num_threads=1,
    )
    return np.asarray(preds, dtype=int), np.asarray(probs, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Path to the baseline TFLite model.")
    parser.add_argument("--challenger", type=Path, required=True,
                        help="Path to the challenger TFLite model.")
    parser.add_argument("--baseline-name", type=str, default=None)
    parser.add_argument("--challenger-name", type=str, default=None)
    parser.add_argument("--x", type=Path,
                        default=DEFAULT_RESULTS / "x_test_originals_96_pm1.npy")
    parser.add_argument("--y", type=Path,
                        default=DEFAULT_RESULTS / "y_test_originals.npy")
    parser.add_argument("--capture-ids", type=Path,
                        default=DEFAULT_RESULTS / "capture_ids_originals.npy")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 1
    if not args.challenger.exists():
        print(f"challenger not found: {args.challenger}", file=sys.stderr)
        return 1
    if not args.x.exists() or not args.y.exists():
        print("x/y arrays not found; run python/bench/build_originals_test.py first",
              file=sys.stderr)
        return 1

    baseline_name = args.baseline_name or args.baseline.stem
    challenger_name = args.challenger_name or args.challenger.stem

    x = np.load(args.x)
    y = np.load(args.y)
    capture_ids = (
        np.load(args.capture_ids, allow_pickle=True).astype(str)
        if args.capture_ids.exists() else np.arange(y.shape[0]).astype(str)
    )

    n = y.size
    print(f"Evaluating {baseline_name} on {n} originals...")
    pred_b, probs_b = evaluate(args.baseline, x, y)
    print(f"Evaluating {challenger_name} on {n} originals...")
    pred_c, probs_c = evaluate(args.challenger, x, y)

    correct_b = (pred_b == y).astype(int)
    correct_c = (pred_c == y).astype(int)

    acc_b = float(correct_b.mean())
    acc_c = float(correct_c.mean())
    wilson_b = wilson_ci(int(correct_b.sum()), n)
    wilson_c = wilson_ci(int(correct_c.sum()), n)

    f1_b = macro_f1(y, pred_b)
    f1_c = macro_f1(y, pred_c)
    boot_f1_b = cluster_bootstrap_macro_f1(y, pred_b, capture_ids,
                                           n_boot=args.n_boot, seed=args.seed)
    boot_f1_c = cluster_bootstrap_macro_f1(y, pred_c, capture_ids,
                                           n_boot=args.n_boot, seed=args.seed)

    a = int(np.sum((correct_b == 1) & (correct_c == 1)))
    b_count = int(np.sum((correct_b == 1) & (correct_c == 0)))
    c_count = int(np.sum((correct_b == 0) & (correct_c == 1)))
    d = int(np.sum((correct_b == 0) & (correct_c == 0)))

    p_value = exact_mcnemar(b_count, c_count)

    discordant_lines: list[str] = []
    for idx in range(n):
        if correct_b[idx] != correct_c[idx]:
            discordant_lines.append(
                f"| {capture_ids[idx]} | {label_name(int(y[idx]))} | "
                f"{label_name(int(pred_b[idx]))} | {label_name(int(pred_c[idx]))} |"
            )

    delta_acc = acc_c - acc_b
    delta_f1 = f1_c - f1_b

    significant_acc = abs(delta_acc) >= 0.02 and p_value < 0.05
    significant_f1 = abs(delta_f1) >= 0.02 and p_value < 0.05

    if significant_f1:
        winner = challenger_name if delta_f1 > 0 else baseline_name
        verdict = (f"Per the lexicographic decision rule (Δ macro-F1 ≥ 0.02 AND exact "
                   f"McNemar p < 0.05), **{winner}** is preferred on this honest "
                   f"originals-only test.")
    else:
        verdict = ("Lexicographic decision rule NOT met "
                   "(Δ macro-F1 < 0.02 or McNemar p ≥ 0.05). "
                   "Tie-breakers: TFLite size → MAC count → maturity (manual). "
                   "Final pick is signed off by the human team in `docs/decision.md`.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Head-to-head: `{baseline_name}` vs `{challenger_name}`",
        "",
        f"- Test set: {n} originals-only captures (cleaned of augmented variants).",
        f"- Predictions, capture-level pairing.",
        f"- Bootstrap n_boot={args.n_boot}, seed={args.seed}.",
        "",
        "## Per-model headline",
        "",
        "| model | correct/n | accuracy | Wilson 95% CI | macro-F1 | bootstrap macro-F1 95% CI |",
        "|---|---:|---:|---|---:|---|",
        (f"| {baseline_name} | {int(correct_b.sum())}/{n} | {acc_b*100:.2f}% | "
         f"[{wilson_b[0]*100:.2f}%, {wilson_b[1]*100:.2f}%] | {f1_b:.4f} | "
         f"[{boot_f1_b[1]:.4f}, {boot_f1_b[2]:.4f}] |"),
        (f"| {challenger_name} | {int(correct_c.sum())}/{n} | {acc_c*100:.2f}% | "
         f"[{wilson_c[0]*100:.2f}%, {wilson_c[1]*100:.2f}%] | {f1_c:.4f} | "
         f"[{boot_f1_c[1]:.4f}, {boot_f1_c[2]:.4f}] |"),
        "",
        f"Δaccuracy = {delta_acc*100:+.2f} pp",
        f"Δmacro-F1 = {delta_f1:+.4f}",
        "",
        "## Paired McNemar table (capture-level)",
        "",
        f"|  | {challenger_name} correct | {challenger_name} wrong |",
        "|---|---:|---:|",
        f"| {baseline_name} correct | a = {a} | b = {b_count} |",
        f"| {baseline_name} wrong | c = {c_count} | d = {d} |",
        "",
        f"Discordant pairs: b + c = {b_count + c_count}",
        f"Exact two-sided McNemar p-value: **{p_value:.4f}**",
        "",
        "## Discordant captures",
        "",
    ]

    if discordant_lines:
        lines.append("| capture | true | baseline pred | challenger pred |")
        lines.append("|---|---|---|---|")
        lines.extend(discordant_lines)
    else:
        lines.append("Both models agreed on every capture.")

    lines.extend([
        "",
        "## Decision",
        "",
        verdict,
        "",
        "## Caveats",
        "",
        "- n=60 captures; the asymptotic chi-square McNemar is unsafe at this n. "
        "We use `binomtest(min(b,c), n=b+c, p=0.5, two-sided)` as the exact form.",
        "- Cluster bootstrap resamples captures, not file variants, so each capture "
        "contributes once per draw. The 95% CIs are wide as expected for small n.",
        "- Both models are evaluated on the same 60 captures; pairing is at the "
        "capture level, which is the unit of independence.",
        "",
        f"Generated by `python/bench/compare_models.py`.",
    ])

    args.report.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.report}")
    print(f"  {baseline_name}: {acc_b*100:.2f}% acc, F1={f1_b:.4f}")
    print(f"  {challenger_name}: {acc_c*100:.2f}% acc, F1={f1_c:.4f}")
    print(f"  ΔF1 = {delta_f1:+.4f}, McNemar p = {p_value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
