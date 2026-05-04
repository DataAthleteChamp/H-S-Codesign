#!/usr/bin/env python3
"""Distill Keras-Tuner trials into ranked CSVs and a Markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCORE_TOLERANCE = 0.01  # one percentage point of accuracy
HP_FIELDS = (
    "alpha",
    "dense_units",
    "dropout_1",
    "dropout_2",
    "learning_rate",
    "label_smoothing",
)
CSV_FIELDS = (
    "trial_id",
    "score",
    *HP_FIELDS,
    "tuner_bracket",
    "tuner_round",
    "tuner_epochs",
    "tuner_initial_epoch",
    "best_step",
)


@dataclass(frozen=True)
class Trial:
    trial_id: str
    score: float
    alpha: float
    dense_units: int
    dropout_1: float
    dropout_2: float
    learning_rate: float
    label_smoothing: float
    tuner_bracket: int | None
    tuner_round: int | None
    tuner_epochs: int | None
    tuner_initial_epoch: int | None
    best_step: int | None

    def as_row(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in CSV_FIELDS}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Rank Keras-Tuner face-recognition trials and summarize trends."
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=root / "python" / "gen" / "tuner" / "face_recognition",
        help="Directory containing trial_*/trial.json outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "bench" / "results",
        help="Directory where CSV and Markdown outputs will be written.",
    )
    return parser.parse_args()


def load_trials(trials_dir: Path) -> list[Trial]:
    trials: list[Trial] = []
    for trial_json in sorted(trials_dir.glob("trial_*/trial.json")):
        data = json.loads(trial_json.read_text())
        if data.get("status") != "COMPLETED":
            continue

        values = data.get("hyperparameters", {}).get("values", {})
        score = data.get("score")
        if score is None:
            score = latest_metric(data, "val_accuracy")
        if score is None:
            continue

        trial_id = str(
            data.get("trial_id")
            or data.get("id")
            or trial_json.parent.name.removeprefix("trial_")
        )
        trials.append(
            Trial(
                trial_id=trial_id,
                score=float(score),
                alpha=float(values["alpha"]),
                dense_units=int(values["dense_units"]),
                dropout_1=float(values["dropout_1"]),
                dropout_2=float(values["dropout_2"]),
                learning_rate=float(values["learning_rate"]),
                label_smoothing=float(values["label_smoothing"]),
                tuner_bracket=optional_int(values.get("tuner/bracket")),
                tuner_round=optional_int(values.get("tuner/round")),
                tuner_epochs=optional_int(values.get("tuner/epochs")),
                tuner_initial_epoch=optional_int(values.get("tuner/initial_epoch")),
                best_step=optional_int(data.get("best_step")),
            )
        )

    return sorted(trials, key=lambda trial: (-trial.score, trial.alpha, trial.trial_id))


def latest_metric(data: dict[str, object], metric_name: str) -> float | None:
    metric = (
        data.get("metrics", {})
        .get("metrics", {})
        .get(metric_name, {})
    )
    observations = metric.get("observations", []) if isinstance(metric, dict) else []
    if not observations:
        return None
    latest = observations[-1].get("value", [])
    return float(latest[0]) if latest else None


def optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def write_csv(path: Path, trials: Iterable[Trial]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial.as_row())


def pareto_trials(trials: list[Trial]) -> list[Trial]:
    pareto: list[Trial] = []
    for candidate in trials:
        dominated = False
        for challenger in trials:
            if challenger is candidate:
                continue
            smaller_or_equal = challenger.alpha <= candidate.alpha
            matches_score = challenger.score >= candidate.score - SCORE_TOLERANCE
            strictly_better = challenger.alpha < candidate.alpha or challenger.score > candidate.score
            if smaller_or_equal and matches_score and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return sorted(pareto, key=lambda trial: (trial.alpha, -trial.score, trial.trial_id))


def best_by_alpha(trials: list[Trial]) -> dict[float, Trial]:
    result: dict[float, Trial] = {}
    for trial in trials:
        if trial.alpha not in result or trial.score > result[trial.alpha].score:
            result[trial.alpha] = trial
    return dict(sorted(result.items()))


def smallest_within_best(trials: list[Trial]) -> Trial:
    best_score = trials[0].score
    eligible = [trial for trial in trials if trial.score >= best_score - SCORE_TOLERANCE]
    return sorted(eligible, key=lambda trial: (trial.alpha, -trial.score, trial.trial_id))[0]


def group_stats(trials: list[Trial], field: str) -> list[tuple[object, int, float, float, float, str]]:
    grouped: dict[object, list[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[getattr(trial, field)].append(trial)

    rows = []
    for value, group in grouped.items():
        scores = [trial.score for trial in group]
        best = max(group, key=lambda trial: trial.score)
        rows.append((value, len(group), statistics.mean(scores), min(scores), max(scores), best.trial_id))
    return sorted(rows, key=lambda row: row[0])


def percent(score: float) -> str:
    return f"{score * 100:.2f}%"


def decimal(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def config_text(trial: Trial) -> str:
    return (
        f"alpha={trial.alpha:g}, dense_units={trial.dense_units}, "
        f"dropout_1={trial.dropout_1:g}, dropout_2={trial.dropout_2:g}, "
        f"learning_rate={trial.learning_rate:g}, label_smoothing={trial.label_smoothing:g}, "
        f"epochs={trial.tuner_epochs}, bracket={trial.tuner_bracket}, round={trial.tuner_round}"
    )


def markdown_table(headers: tuple[str, ...], rows: Iterable[Iterable[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def trend_sentence(field: str, rows: list[tuple[object, int, float, float, float, str]]) -> str:
    best_mean = max(rows, key=lambda row: row[2])
    best_peak = max(rows, key=lambda row: row[4])
    ordered = ", ".join(
        f"{decimal(value)}: n={count}, mean={percent(mean)}, range={percent(min_score)}–{percent(max_score)}"
        for value, count, mean, min_score, max_score, _ in rows
    )
    return (
        f"- `{field}`: {ordered}. Best average is `{decimal(best_mean[0])}`; "
        f"best single trial is `{decimal(best_peak[0])}` (trial {best_peak[5]} at {percent(best_peak[4])})."
    )


def make_summary(trials: list[Trial], pareto: list[Trial]) -> str:
    best = trials[0]
    efficient_small = smallest_within_best(trials)
    by_alpha = best_by_alpha(trials)
    pareto_ids = {trial.trial_id for trial in pareto}

    top_rows = [
        (
            trial.trial_id,
            percent(trial.score),
            decimal(trial.alpha),
            trial.dense_units,
            decimal(trial.dropout_1),
            decimal(trial.dropout_2),
            decimal(trial.learning_rate),
            decimal(trial.label_smoothing),
            trial.tuner_epochs,
        )
        for trial in trials[:10]
    ]
    alpha_rows = [
        (
            decimal(alpha),
            trial.trial_id,
            percent(trial.score),
            trial.dense_units,
            decimal(trial.learning_rate),
            decimal(trial.label_smoothing),
            "yes" if trial.trial_id in pareto_ids else "no, within 1pp dominated by smaller alpha",
        )
        for alpha, trial in by_alpha.items()
    ]
    pareto_rows = [
        (
            trial.trial_id,
            percent(trial.score),
            decimal(trial.alpha),
            trial.dense_units,
            decimal(trial.learning_rate),
            decimal(trial.label_smoothing),
            trial.tuner_epochs,
        )
        for trial in pareto
    ]

    trends = [
        trend_sentence("alpha", group_stats(trials, "alpha")),
        trend_sentence("dense_units", group_stats(trials, "dense_units")),
        trend_sentence("learning_rate", group_stats(trials, "learning_rate")),
        trend_sentence("label_smoothing", group_stats(trials, "label_smoothing")),
        trend_sentence("dropout_1", group_stats(trials, "dropout_1")),
        trend_sentence("dropout_2", group_stats(trials, "dropout_2")),
    ]

    dense_stats = {row[0]: row for row in group_stats(trials, "dense_units")}
    smoothing_stats = {row[0]: row for row in group_stats(trials, "label_smoothing")}
    dense_note = compare_means(dense_stats, 128, 32, "dense_units=128", "dense_units=32")
    smoothing_note = compare_means(
        smoothing_stats, 0.0, 0.1, "label_smoothing=0", "label_smoothing=0.1"
    )

    recommendation = efficient_small
    if best.alpha <= efficient_small.alpha and best.score - efficient_small.score > SCORE_TOLERANCE:
        recommendation = best

    lines = [
        "# Keras-Tuner trial distillation",
        "",
        f"Parsed **{len(trials)} completed trials** from `python/gen/tuner/face_recognition/trial_*/trial.json`.",
        "Scores are `val_accuracy` from the tuner run.",
        "",
        "## Best trial",
        "",
        f"Best score: **{percent(best.score)}** (trial `{best.trial_id}`) with `{config_text(best)}`.",
        "",
        "## Top 10 trials",
        "",
        markdown_table(
            ("trial", "score", "alpha", "dense", "drop1", "drop2", "lr", "smooth", "epochs"),
            top_rows,
        ),
        "",
        "## Smallest model within 1 percentage point of best",
        "",
        (
            f"The smallest alpha within 1pp of the best score is **alpha={efficient_small.alpha:g}** "
            f"(trial `{efficient_small.trial_id}`, {percent(efficient_small.score)}; "
            f"{config_text(efficient_small)})."
        ),
        "",
        "## Alpha tradeoffs",
        "",
        markdown_table(
            ("alpha", "best trial", "score", "dense", "lr", "smooth", "1pp Pareto?"),
            alpha_rows,
        ),
        "",
        "## 1pp Pareto frontier (score vs alpha)",
        "",
        markdown_table(
            ("trial", "score", "alpha", "dense", "lr", "smooth", "epochs"),
            pareto_rows,
        ),
        "",
        "## Hyperparameter trends",
        "",
        *trends,
        f"- Dense-width check: {dense_note}",
        f"- Label-smoothing check: {smoothing_note}",
        "",
        (
            "These averages mix Hyperband brackets/epoch budgets, so they are trend indicators "
            "rather than controlled ablations."
        ),
        "",
        "## Recommendation for Phase 2 retraining",
        "",
        (
            f"Use trial `{recommendation.trial_id}` as the **jakubs candidate** starting point: "
            f"`{config_text(recommendation)}`. It is the smallest-alpha configuration within 1pp "
            "of the best observed score, so it preserves nearly all tuner accuracy while reducing "
            "the MobileNetV2 backbone width."
        ),
        "",
        "## Caveat",
        "",
        (
            "The tuner used the biased validation set, which is the documented leaked test set. "
            "Therefore these absolute `val_accuracy` numbers must not be quoted as final performance. "
            "Expect scores to drop by roughly 3–5 percentage points on the cleaned originals-only test set; "
            "the ranking should still be useful as design-space exploration evidence."
        ),
        "",
    ]
    return "\n".join(lines)


def compare_means(
    stats: dict[object, tuple[object, int, float, float, float, str]],
    a: object,
    b: object,
    a_name: str,
    b_name: str,
) -> str:
    if a not in stats or b not in stats:
        return "insufficient coverage for a direct mean comparison."
    a_mean = stats[a][2]
    b_mean = stats[b][2]
    a_best = stats[a][4]
    b_best = stats[b][4]
    delta_pp = (a_mean - b_mean) * 100
    best_delta_pp = (a_best - b_best) * 100

    if abs(delta_pp) < 0.25:
        mean_part = (
            f"{a_name} and {b_name} are essentially tied on mean "
            f"({abs(delta_pp):.2f}pp)."
        )
    elif delta_pp > 0:
        mean_part = f"{a_name} has a higher mean than {b_name} by {abs(delta_pp):.2f}pp."
    else:
        mean_part = f"{a_name} has a lower mean than {b_name} by {abs(delta_pp):.2f}pp."

    if abs(best_delta_pp) < 0.05:
        peak_part = "Their best single-trial scores are effectively tied."
    elif best_delta_pp > 0:
        peak_part = f"{a_name} also has the better best single trial by {abs(best_delta_pp):.2f}pp."
    else:
        peak_part = f"However, {b_name} has the better best single trial by {abs(best_delta_pp):.2f}pp."

    return f"{mean_part} {peak_part} Treat this as a trend, not a dominance proof."


def main() -> None:
    args = parse_args()
    trials = load_trials(args.trials_dir)
    if not trials:
        raise SystemExit(f"No completed trials found under {args.trials_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    pareto = pareto_trials(trials)

    write_csv(args.results_dir / "tuner_all.csv", trials)
    write_csv(args.results_dir / "tuner_top10.csv", trials[:10])
    write_csv(args.results_dir / "tuner_pareto.csv", pareto)

    summary = make_summary(trials, pareto)
    (args.results_dir / "tuner_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
