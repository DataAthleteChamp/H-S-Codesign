"""Statistical helpers for Phase 3 face-recognition evaluation.

All confidence intervals operate at the original-capture unit. For a future
head-to-head comparison, the paired permutation robustness check should flip the
two candidate predictions within discordant capture pairs and recompute the
paired metric difference; it is deferred while only one model is evaluated.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, norm

SWEEP_DTYPE = np.dtype(
    [
        ("threshold", "f8"),
        ("n_accepted", "i8"),
        ("accept_rate", "f8"),
        ("accuracy_on_accepted", "f8"),
        ("ece_on_accepted", "f8"),
    ]
)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for k successes out of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half_width = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    """Compute macro-F1 without sklearn, using zero_division=0 semantics."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true/y_pred length mismatch")
    if labels is None:
        labels = np.union1d(y_true, y_pred)
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return float("nan")

    scores = []
    for label in labels:
        true_pos = np.sum((y_true == label) & (y_pred == label))
        false_pos = np.sum((y_true != label) & (y_pred == label))
        false_neg = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * true_pos + false_pos + false_neg
        scores.append(0.0 if denom == 0 else (2.0 * true_pos) / float(denom))
    return float(np.mean(scores))


def cluster_bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capture_ids: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return observed macro-F1 and percentile 95% CI from capture bootstrap."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    capture_ids = np.asarray(capture_ids).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != capture_ids.shape[0]:
        raise ValueError("y_true, y_pred, and capture_ids must have the same length")
    if y_true.size == 0:
        raise ValueError("at least one sample is required")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    labels = np.union1d(y_true, y_pred)
    point = macro_f1(y_true, y_pred, labels=labels)
    _, inverse = np.unique(capture_ids.astype(str), return_inverse=True)
    clusters = [np.flatnonzero(inverse == idx) for idx in range(int(inverse.max()) + 1)]
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)

    for idx in range(int(n_boot)):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sample_indices = np.concatenate([clusters[item] for item in selected])
        boot[idx] = macro_f1(y_true[sample_indices], y_pred[sample_indices], labels=labels)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(confidence.size)
    ece = 0.0
    for idx in range(int(n_bins)):
        if idx == int(n_bins) - 1:
            mask = (confidence >= edges[idx]) & (confidence <= edges[idx + 1])
        else:
            mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1])
        if np.any(mask):
            ece += (np.sum(mask) / total) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(ece)


def rejection_sweep(
    softmax_probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.0, 0.99, 100),
) -> np.ndarray:
    """Return a structured table for confidence-threshold rejection analysis."""
    probs = np.asarray(softmax_probs, dtype=np.float64)
    y_true = np.asarray(y_true).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("softmax_probs must have shape (n_samples, n_classes)")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("softmax_probs/y_true length mismatch")

    y_pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    rows = np.empty(thresholds.shape[0], dtype=SWEEP_DTYPE)

    for idx, threshold in enumerate(thresholds):
        accepted = confidence >= threshold
        n_accepted = int(np.sum(accepted))
        accept_rate = n_accepted / float(y_true.size) if y_true.size else float("nan")
        if n_accepted:
            accepted_correct = y_pred[accepted] == y_true[accepted]
            acc = float(np.mean(accepted_correct))
            ece = _ece(confidence[accepted], accepted_correct) if n_accepted >= 10 else float("nan")
        else:
            acc = float("nan")
            ece = float("nan")
        rows[idx] = (float(threshold), n_accepted, accept_rate, acc, ece)
    return rows


def exact_mcnemar(b: int, c: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant counts b and c."""
    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    total = b + c
    if total == 0:
        return 1.0
    return float(binomtest(min(b, c), n=total, p=0.5, alternative="two-sided").pvalue)
