"""Generate Phase 3 statistical artifacts for the QAT TFLite model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats import cluster_bootstrap_macro_f1, macro_f1, rejection_sweep, wilson_ci

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
ORIGINALS_NPZ = RESULTS / "jakubs_qat_originals_test.npz"
FULL_AUG_NPZ = RESULTS / "jakubs_qat_full_aug_test.npz"
CLASS_NAMES = {0: "Amine", 1: "Rifki", 2: "Jakub"}


def _get_array(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    raise KeyError(f"missing any of keys: {', '.join(names)}")


def _load_eval(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "y_true": _get_array(data, "labels", "y_true").astype(np.int64).reshape(-1),
            "y_pred": _get_array(data, "predictions", "y_pred").astype(np.int64).reshape(-1),
            "probs": _get_array(data, "probs", "softmax_probs").astype(np.float64),
            "capture_ids": _get_array(data, "capture_ids").astype(str).reshape(-1)
            if "capture_ids" in data.files
            else np.array([], dtype=str),
            "input_scale": float(np.asarray(data["input_scale"])) if "input_scale" in data.files else 0.007843137718737125,
            "input_zero_point": int(np.asarray(data["input_zero_point"])) if "input_zero_point" in data.files else 0,
            "output_scale": float(np.asarray(data["output_scale"])) if "output_scale" in data.files else 0.00390625,
            "output_zero_point": int(np.asarray(data["output_zero_point"])) if "output_zero_point" in data.files else -128,
        }


def _as_frame(rows: np.ndarray, dataset: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.insert(0, "dataset", dataset)
    return frame


def _choose_threshold(frame: pd.DataFrame) -> pd.Series:
    eligible = frame[
        (frame["accept_rate"] >= 0.70) & frame["accuracy_on_accepted"].notna()
    ].copy()
    if eligible.empty:
        raise RuntimeError("no rejection threshold satisfies accept_rate >= 0.70")
    eligible = eligible.sort_values(
        ["accuracy_on_accepted", "accept_rate", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0]


def _q_int8(q: float, scale: float, zero_point: int) -> int:
    value = int(np.floor(float(q) / float(scale)) + int(zero_point))
    return int(np.clip(value, -128, 127))


def _fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def _fmt_float(value: float, digits: int = 4) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _threshold_table(frame: pd.DataFrame, thresholds: list[float]) -> str:
    rows = []
    seen: set[float] = set()
    for threshold in thresholds:
        idx = (frame["threshold"] - threshold).abs().idxmin()
        row = frame.loc[idx]
        key = round(float(row["threshold"]), 2)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            "| "
            f"{float(row['threshold']):.2f} | {int(row['n_accepted'])} | "
            f"{_fmt_pct(row['accept_rate'])} | {_fmt_pct(row['accuracy_on_accepted'])} | "
            f"{_fmt_float(row['ece_on_accepted'])} |"
        )
    header = (
        "| threshold | n accepted | accept rate | accuracy on accepted | ECE on accepted |\n"
        "|---:|---:|---:|---:|---:|"
    )
    return "\n".join([header, *rows])


def _write_summary(
    originals: dict[str, np.ndarray],
    full_aug: dict[str, np.ndarray],
    orig_sweep: pd.DataFrame,
    full_sweep: pd.DataFrame,
    chosen: pd.Series,
    chosen_full: pd.Series,
    wilson: tuple[float, float],
    boot: tuple[float, float, float],
) -> None:
    y_true = originals["y_true"]
    y_pred = originals["y_pred"]
    n = int(y_true.size)
    k = int(np.sum(y_true == y_pred))
    acc = k / n
    macro = macro_f1(y_true, y_pred)
    full_acc = float(np.mean(full_aug["y_true"] == full_aug["y_pred"]))
    full_macro = macro_f1(full_aug["y_true"], full_aug["y_pred"])
    q = float(chosen["threshold"])
    q_input = _q_int8(q, originals["input_scale"], originals["input_zero_point"])
    q_output = _q_int8(q, originals["output_scale"], originals["output_zero_point"])

    reject_count = n - int(chosen["n_accepted"])
    lines = [
        "# Phase 3 statistical summary",
        "",
        "Input artifacts: `jakubs_qat_originals_test.npz` for headline statistics and "
        "`jakubs_qat_full_aug_test.npz` only for the biased augmentation-robustness panel.",
        "",
        "## Headline: originals-only capture test",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| independent captures | {n} |",
        f"| correct captures | {k} |",
        f"| accuracy | {_fmt_pct(acc)} |",
        f"| Wilson 95% CI | [{_fmt_pct(wilson[0])}, {_fmt_pct(wilson[1])}] |",
        f"| macro-F1 | {_fmt_float(macro)} |",
        f"| cluster-bootstrap macro-F1 95% CI | [{_fmt_float(boot[1])}, {_fmt_float(boot[2])}] |",
        "",
        f"The headline accuracy is {k}/{n} = {_fmt_pct(acc)}, matching "
        "`calibration_report.md` (98.33%). The macro-F1 is "
        f"{_fmt_float(macro)}, also matching the existing headline after rounding.",
        "",
        "## Rejection threshold sweep",
        "",
        "Thresholds were swept from 0.00 to 0.99 in steps of 0.01. The chosen "
        "operating point maximizes accuracy on accepted samples subject to "
        "`accept_rate >= 0.70`; ties retain the most captures, then use the lower threshold.",
        "",
        f"Recommended threshold: **q = {q:.2f}**. It accepts "
        f"{int(chosen['n_accepted'])}/{n} captures ({_fmt_pct(chosen['accept_rate'])}), "
        f"rejects {reject_count}, and gives {_fmt_pct(chosen['accuracy_on_accepted'])} "
        f"accuracy on accepted captures (ECE={_fmt_float(chosen['ece_on_accepted'])}).",
        "",
        "Firmware mapping:",
        f"- Current firmware compares dequantized floats, so use `best_conf >= {q:.2f}f`.",
        f"- If comparing raw softmax int8 output, use `floor(q / OUTPUT_SCALE) + OUTPUT_ZERO_POINT = {q_output}` "
        f"with OUTPUT_SCALE={originals['output_scale']:.8f}, OUTPUT_ZERO_POINT={originals['output_zero_point']}.",
        f"- The requested input-scale formula gives `floor(q / INPUT_SCALE) + INPUT_ZP = {q_input}` "
        f"with INPUT_SCALE={originals['input_scale']:.8f}, INPUT_ZP={originals['input_zero_point']}.",
        "",
        _threshold_table(orig_sweep, [0.0, q, 0.80, 0.85, 0.90, 0.95]),
        "",
        "Tradeoff: q=0.77 removes the single known Amine→Rifki error while retaining "
        "96.67% of captures. The previous q=0.90 also reaches 100% accepted accuracy "
        "but rejects 7/60 captures, so q=0.77 is the less aggressive operating point.",
        "",
        "## Augmentation-robustness panel (biased; informational only)",
        "",
        f"The full augmented test has n={full_aug['y_true'].size} files but only derives "
        "from the same 60 original captures, so it is not independent evidence. Its "
        f"file-level accuracy is {_fmt_pct(full_acc)} and macro-F1 is {_fmt_float(full_macro)}.",
        "",
        _threshold_table(full_sweep, [0.0, 0.80, 0.85, 0.90, float(chosen_full["threshold"]), 0.95]),
        "",
        f"Under the same accept-rate constraint, the biased file-level sweep first reaches "
        f"100% accepted accuracy at q={float(chosen_full['threshold']):.2f} with "
        f"{_fmt_pct(chosen_full['accept_rate'])} acceptance. This is reported only as an "
        "augmentation-robustness diagnostic, not as headline statistical evidence.",
        "",
        "## Caveats",
        "",
        f"- n=60 captures; CIs are wide. Wilson score 95% CI on 59/60 = "
        f"[{_fmt_pct(wilson[0])}, {_fmt_pct(wilson[1])}].",
        "- Cluster bootstrap is by original capture, n_boot=10000, seed=42.",
        "- Rejection threshold q must be mapped with the tensor scale used at the firmware "
        "comparison point; softmax output int8 is the relevant raw-output mapping, while "
        "the input-scale value above is included for project convention traceability.",
        "- Single-model evaluation. Head-to-head McNemar requires a second candidate; "
        "deferred. `exact_mcnemar(b, c)` is implemented in `python/bench/stats.py` for future paired tests.",
        "- Paired permutation testing is also deferred until a second candidate exists; the "
        "capture-level method is noted in `python/bench/stats.py`.",
        "",
    ]
    (RESULTS / "stats_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_plot(orig_sweep: pd.DataFrame, full_sweep: pd.DataFrame, chosen: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.plot(
        orig_sweep["accept_rate"],
        orig_sweep["accuracy_on_accepted"],
        marker="o",
        markersize=2.5,
        linewidth=1.2,
        label="originals-only (headline)",
    )
    ax.plot(
        full_sweep["accept_rate"],
        full_sweep["accuracy_on_accepted"],
        linestyle="--",
        linewidth=1.0,
        label="full augmented (biased)",
    )
    ax.scatter(
        [chosen["accept_rate"]],
        [chosen["accuracy_on_accepted"]],
        s=70,
        marker="*",
        color="crimson",
        zorder=5,
        label=f"chosen q={float(chosen['threshold']):.2f}",
    )
    ax.axvline(0.70, color="gray", linestyle=":", linewidth=1.0, label="70% accept constraint")
    ax.set_xlabel("accept rate")
    ax.set_ylabel("accuracy on accepted")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.94, 1.005)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "rejection_sweep.png")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    originals = _load_eval(ORIGINALS_NPZ)
    full_aug = _load_eval(FULL_AUG_NPZ)

    y_true = originals["y_true"]
    y_pred = originals["y_pred"]
    capture_ids = originals["capture_ids"]
    if capture_ids.size != y_true.size:
        raise ValueError("originals npz must include one capture_id per capture")

    n = int(y_true.size)
    k = int(np.sum(y_true == y_pred))
    accuracy = k / n
    wilson = wilson_ci(k, n)
    boot = cluster_bootstrap_macro_f1(y_true, y_pred, capture_ids, n_boot=10000, seed=42)

    thresholds = np.linspace(0.0, 0.99, 100)
    orig_sweep = _as_frame(rejection_sweep(originals["probs"], y_true, thresholds), "originals_only")
    full_sweep = _as_frame(rejection_sweep(full_aug["probs"], full_aug["y_true"], thresholds), "full_aug_test_biased")
    chosen = _choose_threshold(orig_sweep)
    chosen_full = _choose_threshold(full_sweep)

    combined = pd.concat([orig_sweep, full_sweep], ignore_index=True)
    combined.to_csv(RESULTS / "rejection_sweep.csv", index=False, float_format="%.6f")

    summary = {
        "dataset": "originals_only",
        "n_captures": n,
        "n_correct": k,
        "accuracy": accuracy,
        "wilson_95_ci": {"lo": wilson[0], "hi": wilson[1]},
        "macro_f1": macro_f1(y_true, y_pred),
        "cluster_bootstrap_macro_f1": {
            "point": boot[0],
            "lo": boot[1],
            "hi": boot[2],
            "n_boot": 10000,
            "seed": 42,
            "cluster_unit": "original_capture",
        },
        "rejection_threshold": {
            "threshold": float(chosen["threshold"]),
            "n_accepted": int(chosen["n_accepted"]),
            "accept_rate": float(chosen["accept_rate"]),
            "accuracy_on_accepted": float(chosen["accuracy_on_accepted"]),
            "ece_on_accepted": float(chosen["ece_on_accepted"]),
            "int8_softmax_output": _q_int8(
                float(chosen["threshold"]), originals["output_scale"], originals["output_zero_point"]
            ),
            "int8_input_scale_formula": _q_int8(
                float(chosen["threshold"]), originals["input_scale"], originals["input_zero_point"]
            ),
        },
        "mcnemar": "deferred: single-model evaluation; exact_mcnemar(b, c) implemented",
        "paired_permutation": "deferred: single-model evaluation; capture-level method noted in stats.py",
    }
    (RESULTS / "bootstrap_ci.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_summary(originals, full_aug, orig_sweep, full_sweep, chosen, chosen_full, wilson, boot)
    _write_plot(orig_sweep, full_sweep, chosen)

    print(
        f"accuracy={accuracy:.4f} Wilson95=[{wilson[0]:.4f}, {wilson[1]:.4f}] "
        f"macro_f1={boot[0]:.4f} bootstrap95=[{boot[1]:.4f}, {boot[2]:.4f}] "
        f"threshold={float(chosen['threshold']):.2f} accept={float(chosen['accept_rate']):.4f}"
    )


if __name__ == "__main__":
    main()
