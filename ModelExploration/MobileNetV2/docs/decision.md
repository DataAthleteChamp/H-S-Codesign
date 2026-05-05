# Branch consolidation decision

> **Status: DRAFT — pending team sign-off.** The course rules state
> *"Letting the AI make design decisions is not allowed."* This document
> presents the evidence and the recommendation framework. The final
> selection is made and signed by the team members named at the end.

## Question

Among the three branches (`amineModel`, `main` = Rifki, `jakubs-solution`
= Jakub), which one becomes the trunk for the May 7 2026 deliverable?

## Evidence summary

### `amineModel`

A stale subset of an earlier `jakubs-solution` snapshot. Lacks the
firmware, the QAT pipeline, the design-space scripts, the tuner trials,
and the calibration/eval harness. Only unique content is an alternate
`python/augment.py` (older Albumentations 1.x API).

**Recommendation: ABANDON.** Tagged `archive/amineModel` (commit
`a375d14`) for traceability, then no further work flows through it.

### `main` (Rifki)

Contains a 160×160 PTQ pipeline with `MobileNetV2_Crop.ipynb` and
`Quantization.ipynb` notebooks, plus the USB-stream firmware feature
that was useful enough to cherry-pick into the trunk during the F3
firmware fix (commit `ff18dcd`).

**Recommendation: ARCHIVE + CHERRY-PICK.** The USB-stream code is
already merged into the trunk. The notebooks remain on `main` and are
referenced from the design-space discussion in the report. Tag
`archive/rifki-main` will be added at the end of Phase 5.

### `jakubs-solution` (Jakub) — proposed trunk

Has the broadest scope and the most evaluation-ready artefacts:

- 96×96 INT8 QAT pipeline (`python/main.py`, `python/qat_export.py`)
- Modern Albumentations 2.0 + MediaPipe Tasks API (`python/preprocess.py`,
  `python/augment.py`)
- 30 completed Keras-Tuner Hyperband trials parsed in
  `bench/results/tuner_summary.md`
- Calibration + statistical harness in `python/bench/`
- F2 fix: held-out validation drawn from train (no test leak), see
  `python/utils/train_val_split.py` and the `cf02a95` commit
- F3 firmware fix already landed (`ff18dcd`)
- Both candidate INT8 TFLite models present:
  - `python/gen/model.tflite` (existing deployed, 662 KB)
  - `python/gen/baseline_model.tflite` (F2-clean retrain, 662 KB)

## Quantitative comparison (originals-only test, n = 60 captures)

See `bench/results/mcnemar_comparison.md` for the full breakdown.

| Metric | existing `model.tflite` | `baseline_model.tflite` (F2-clean) |
|---|---:|---:|
| correct / n | 59 / 60 | 56 / 60 |
| accuracy | 98.33 % | 93.33 % |
| Wilson 95 % CI | [91.14 %, 99.71 %] | [84.07 %, 97.38 %] |
| macro-F1 | 0.9833 | 0.9329 |
| bootstrap macro-F1 95 % CI | [0.9433, 1.0000] | [0.8602, 0.9842] |
| TFLite size | 662 056 B | 662 056 B |
| MAC count | 10.70 M | 10.70 M |

Paired McNemar (capture-level discordants):

|  | existing correct | existing wrong |
|---|---:|---:|
| baseline correct | 56 | 0 |
| baseline wrong | 3 | 1 |

Exact two-sided McNemar p = **0.25** — **not** statistically significant
at α = 0.05.

## Pre-registered lexicographic decision rule

Predeclared in `plan.md` §4 before measurements were taken:

1. Compute macro-F1 on the cleaned originals-only test.
2. Declare a winner only if **both** conditions hold:
   - Δ macro-F1 ≥ 0.02
   - Exact McNemar p < 0.05
3. Otherwise, fall through to tie-breakers in this order:
   - smaller TFLite size,
   - smaller MAC count,
   - manual maturity judgement signed off by the team.

## Application

| Condition | Threshold | Observed | Met? |
|---|---|---|---|
| Δ macro-F1 | ≥ 0.02 | +0.0504 | ✅ |
| McNemar p | < 0.05 | 0.25 | ❌ |

Both conditions are required, so **the rule does not declare a winner**.
Tie-breakers apply.

| Tie-breaker | existing | baseline | Decision |
|---|---:|---:|---|
| TFLite size | 662 056 B | 662 056 B | TIE |
| MAC count | 10.70 M | 10.70 M | TIE |
| Maturity | Already deployed; trained with F2 leak | F2-clean retrain | manual |

## Recommendation framework (evidence-based)

Two reasonable readings:

**Reading A — keep the existing model deployed.**
Point-estimate accuracy is +5 pp higher. The val=test leak biased
*model selection* (early stopping picked the epoch that did best on the
augmented test) but the model still never saw the 60 originals as
training labels, so the leak inflates the score by less than the
typical model-selection bias range (1–3 pp). The deployed model is
already in firmware; switching costs are non-zero.

**Reading B — switch to the F2-clean baseline.**
Methodologically cleaner. The 5 pp gap is not statistically
distinguishable from sampling noise at n = 60 (p = 0.25). The honest
report number 93.33 % still meets any reasonable application
requirement and avoids the optics of disclosing a known leak.

Both are defensible; the team chooses based on the report narrative
the team wants to write. Either choice should explicitly disclose:

- the F2 leak in the existing pipeline,
- the n = 60 honest sample size,
- the McNemar non-significance,
- the chosen model's headline number with its Wilson 95 % CI.

## Team sign-off

This section is filled in **by hand** during the consolidation review.
Do not let the AI fill it in.

- Selected trunk model: `__________________________________`
- Justification (1–3 sentences, course-theory grounded):

  ```
  ___________________________________________________________
  ___________________________________________________________
  ___________________________________________________________
  ```

- Sign-off (one line per team member):

  - Amine: __________________ date: ___________
  - Rifki: __________________ date: ___________
  - Jakub: __________________ date: ___________

## References

- `plan.md` — full plan and pre-registered methodology
- `docs/branch-audit.md` — branch inventory and history
- `bench/results/calibration_report.md` — first honest evaluation
- `bench/results/mcnemar_comparison.md` — head-to-head numbers
- `bench/results/stats_summary.md` — Wilson CI, cluster bootstrap, rejection sweep
- `bench/results/tuner_summary.md` — Phase 4 distillation of 30 tuner trials
- `bench/results/baseline_retrain_report.md` — F2-clean retrain run notes
- `bench/results/firmware_preprocess_check.md` — F3 regression test

## Addendum — F1 on-disk cleanup

The historical contamination of `data/<person>/test/` with 720
augmented variants (240 per class) has been remediated post-hoc. The
headline numbers above are unchanged because the bench harness was
already filtering by suffix; the cleanup is a methodological hygiene
fix that prevents recurrence.

- `bench/results/test_pollution_inventory.md` — pre-cleanup audit with
  SHA-256 of every removed file (720 entries).
- `data/_quarantine/test_augmented/manifest.json` — quarantine manifest
  recording what was moved and where (720 entries).
- `bench/results/postclean_metrics.md` — post-cleanup re-evaluation
  table showing identical metrics; reproduces the table in §4 above.
- `docs/report/methods_test_hygiene.md` — short methods note for the
  report with the academic references.
- `python/tools/clean_test_augmentations.py` — the cleanup tool.
- `python/augment.py` and `python/preprocess.py` carry defensive guards
  so a future re-run cannot silently re-pollute the test split.

