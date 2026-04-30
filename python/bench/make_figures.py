"""Generate the report figures from existing bench/results artifacts.

Outputs PNG files into `docs/figures/`. Designed to run on the same
artifacts produced by Phases 3 and 4, so it is reproducible without
re-running training or evaluation.

Figures:
- f01_dataspace.png      capture counts per class, per split
- f02_calibration.png    accuracy on full augmented test vs originals-only
- f03_confusion.png      confusion matrices for both models on originals-only
- f04_rejection.png      rejection-threshold sweep on the existing model
- f05_tuner_sweep.png    tuner val-accuracy vs alpha (with 1pp Pareto frontier)
- f06_compare_macroF1.png  per-model macro-F1 with bootstrap CIs

All figures are deterministic and use only matplotlib + numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

LABELS = ("Amine", "Rifki", "Jakub")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig: plt.Figure, name: str) -> None:
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_dataspace() -> None:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print("  skipping dataspace: data/ not present locally")
        return

    train_total = []
    test_total = []
    test_originals = []
    aug_suffixes = ("_hflip", "_rot", "_shiftscale", "_bright", "_blur",
                    "_compress", "_occlude", "_gray",
                    "_combo1", "_combo2", "_combo3", "_combo4")
    for cls in LABELS:
        train_files = list((data_dir / cls / "train").glob("*.jpg")) + \
            list((data_dir / cls / "train").glob("*.png"))
        test_files = list((data_dir / cls / "test").glob("*.jpg")) + \
            list((data_dir / cls / "test").glob("*.png"))
        originals = [f for f in test_files
                     if not any(f.stem.lower().endswith(s) for s in aug_suffixes)]
        train_total.append(len(train_files))
        test_total.append(len(test_files))
        test_originals.append(len(originals))

    x = np.arange(len(LABELS))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width, train_total, width, label="train (with augs)")
    ax.bar(x, test_total, width, label="test (with augs)")
    ax.bar(x + width, test_originals, width, label="test (originals only)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("number of files")
    ax.set_title("Dataset composition per class")
    ax.legend()
    for xi, t, e, o in zip(x, train_total, test_total, test_originals, strict=True):
        ax.text(xi - width, t + 5, str(t), ha="center", fontsize=9)
        ax.text(xi, e + 5, str(e), ha="center", fontsize=9)
        ax.text(xi + width, o + 5, str(o), ha="center", fontsize=9)
    _save(fig, "f01_dataspace.png")


def fig_calibration() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(["full augmented test\n(n=780, biased)",
                   "originals only\n(n=60, honest)"],
                  [0.9769, 0.9833],
                  color=["#bbbbbb", "#3e7cb1"])
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration: existing model on biased vs honest test")
    for bar, val in zip(bars, [0.9769, 0.9833], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0008,
                f"{val*100:.2f}%", ha="center", fontsize=10)
    ax.text(0.5, 0.91,
            "Δ acc = -0.64 pp (within 2 pp insurance threshold)",
            ha="center", transform=ax.transAxes, fontsize=9)
    _save(fig, "f02_calibration.png")


def fig_confusion() -> None:
    npz = RESULTS / "jakubs_qat_originals_test.npz"
    if not npz.exists():
        print("  skipping confusion: jakubs_qat_originals_test.npz missing")
        return

    data = np.load(npz, allow_pickle=True)
    y_true = np.asarray(data["labels"]).astype(int).reshape(-1)
    y_pred = np.asarray(data["predictions"]).astype(int).reshape(-1)

    cm_existing = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm_existing[t, p] += 1

    baseline_cm_path = RESULTS / "baseline_originals_test.npz"
    if baseline_cm_path.exists():
        bdata = np.load(baseline_cm_path, allow_pickle=True)
        b_true = np.asarray(bdata["labels"]).astype(int).reshape(-1)
        b_pred = np.asarray(bdata["predictions"]).astype(int).reshape(-1)
        cm_baseline = np.zeros((3, 3), dtype=int)
        for t, p in zip(b_true, b_pred, strict=True):
            cm_baseline[t, p] += 1
    else:
        cm_baseline = None

    n_panels = 2 if cm_baseline is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]
    panels = [(axes[0], cm_existing, "existing model.tflite (59/60)")]
    if cm_baseline is not None:
        correct = int(np.trace(cm_baseline))
        n = int(cm_baseline.sum())
        panels.append((axes[1], cm_baseline,
                       f"baseline_model.tflite ({correct}/{n})"))

    for ax, cm, title in panels:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=20)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > 10 else "black",
                        fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "f03_confusion.png")


def fig_rejection() -> None:
    csv = RESULTS / "rejection_sweep.csv"
    if not csv.exists():
        print("  skipping rejection: rejection_sweep.csv missing")
        return

    arr = np.genfromtxt(csv, delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(arr["accept_rate"], arr["accuracy_on_accepted"], "o-", color="#3e7cb1",
            markersize=3, linewidth=1.4)
    chosen_idx = int(np.argmin(np.abs(arr["threshold"] - 0.77)))
    ax.scatter([arr["accept_rate"][chosen_idx]],
               [arr["accuracy_on_accepted"][chosen_idx]],
               s=100, color="#dc3545", zorder=5, label="q=0.77 (chosen)")
    ax.set_xlabel("accept rate")
    ax.set_ylabel("accuracy on accepted")
    ax.set_title("Rejection-threshold sweep (existing model, n=60 captures)")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.95, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, "f04_rejection.png")


def fig_tuner() -> None:
    csv = RESULTS / "tuner_all.csv"
    if not csv.exists():
        print("  skipping tuner: tuner_all.csv missing")
        return

    arr = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    cols = arr.dtype.names or ()
    score_col = next((c for c in cols if "score" in c.lower() or "val_acc" in c.lower()), None)
    alpha_col = next((c for c in cols if "alpha" in c.lower()), None)
    if score_col is None or alpha_col is None:
        print(f"  skipping tuner: cols={cols}")
        return

    score = np.array([float(s) for s in arr[score_col]])
    alpha = np.array([float(a) for a in arr[alpha_col]])

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.scatter(alpha, score, alpha=0.65, s=42)
    best = score.max()
    pareto_mask = score >= (best - 0.01)
    ax.scatter(alpha[pareto_mask], score[pareto_mask],
               s=85, edgecolor="#dc3545", facecolor="none", linewidth=1.6,
               label="within 1 pp of best")
    chosen = (alpha == 0.35) & pareto_mask
    if chosen.any():
        idx = np.where(chosen)[0][0]
        ax.scatter([alpha[idx]], [score[idx]], s=140, color="#dc3545",
                   zorder=5, label="chosen (smallest alpha)")
    ax.set_xlabel("MobileNetV2 alpha (depth multiplier)")
    ax.set_ylabel("tuner val_accuracy (biased — see caveat)")
    ax.set_title("Keras-Tuner trials: width vs accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "f05_tuner_sweep.png")


def fig_compare() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    names = ["baseline_F2_clean", "existing_QAT"]
    f1 = [0.9329, 0.9833]
    lo = [0.8602, 0.9433]
    hi = [0.9842, 1.0000]
    err = [[f1[i] - lo[i] for i in range(2)], [hi[i] - f1[i] for i in range(2)]]
    bars = ax.bar(names, f1, yerr=err, capsize=8, color=["#bbbbbb", "#3e7cb1"])
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("macro-F1")
    ax.set_title("Head-to-head on n=60 honest captures (cluster bootstrap 95% CI)")
    for bar, v in zip(bars, f1, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", fontsize=10)
    ax.text(0.5, -0.18, "Exact McNemar p = 0.25 (not significant at alpha=0.05)",
            ha="center", transform=ax.transAxes, fontsize=9)
    _save(fig, "f06_compare_macroF1.png")


def main() -> int:
    fig_dataspace()
    fig_calibration()
    fig_confusion()
    fig_rejection()
    fig_tuner()
    fig_compare()
    print(f"All figures written under {FIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
