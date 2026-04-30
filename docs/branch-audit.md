# Branch Audit and Consolidation Rationale

This document records the Phase 1 branch audit for the DTU 02214 face-recognition
project. It is evidence for consolidation, not a design-decision sign-off.

The audit compares the three active branch lines:

- `main` / `origin/main` — Rifki branch.
- `jakubs-solution` / `origin/jakubs-solution` — Jakub branch and current trunk candidate.
- `amineModel` / `origin/amineModel` — Amine branch.

The repository was fetched before the audit. The working branch was then pulled with
`git pull --ff-only origin jakubs-solution`, which returned `Already up to date`.
The comparisons below use remote branch heads after that fetch/pull step.

## 1. Executive conclusion

| Branch | Decision | Rationale |
| --- | --- | --- |
| `amineModel` | **ABANDON** | Stale subset. It diverges before the later firmware, notebooks, preview tooling, QAT/export scripts, benchmark harnesses, and report foundations. Its only code idea with possible reuse value is an alternate `python/augment.py`; the branch itself should not be merged. |
| `main` (Rifki) | **MERGE-IN-PARTS** | Contains useful USB-stream firmware and desktop preview lineage. The USB-stream firmware feature has been cherry-picked into `jakubs-solution`; the rest should be treated as archived evidence unless the human team explicitly selects more pieces. |
| `jakubs-solution` (Jakub) | **KEEP AS TRUNK** | Most mature branch: design-space `main.py`, `qat_export.py`, tuner trials, modern Albumentations 2.0 and MediaPipe Tasks API compatibility, benchmark scripts, report foundations, and a deployed INT8 model with honest 98.33% originals-only accuracy. |

The recommended consolidation direction is therefore:

1. Keep `jakubs-solution` as the trunk candidate.
2. Retain selected Rifki functionality already cherry-picked from `main`, especially USB streaming.
3. Do not merge `amineModel`; preserve it by tag only.
4. Let the human team make and sign the final design decision in `docs/decision.md` during Phase 5.

## 2. Branch inventory

| Branch/ref audited | Human label | Last commit | Commit date | Subject |
| --- | --- | --- | --- | --- |
| `origin/main` | Rifki | `b4a11de` | 2026-04-30 13:59:18 +02:00 | `edit preprocess, add camera preview python program` |
| `origin/jakubs-solution` | Jakub | `2c833e6` | 2026-04-30 15:22:26 +02:00 | `fix(firmware): F3 - use MobileNetV2 [-1,1] preprocessing; add USB stream` |
| `origin/amineModel` | Amine | `a375d14` | 2026-04-09 13:39:49 +02:00 | `changed the data augmentation` |

Local note: the stale local `main` branch points at `0a2c6cb`, but this audit uses
`origin/main` because the remote branch is the current Rifki branch head.

## 3. Archive tags

| Tag | Commit | Purpose |
| --- | --- | --- |
| `archive/pre-consolidation` | `68ecb41` | Preserves the pre-consolidation Jakub firmware/pipeline state before later Phase 3/4 work. |
| `archive/amineModel` | `a375d14` | Preserves the abandoned Amine branch head. |

Planned Phase 5 tags:

- `archive/jakubs-solution` will be added during the Phase 5 PR merge.
- `archive/rifki-main` will be added during the Phase 5 PR merge.

## 4. Git evidence snapshot

The required all-branch log command produced the following snapshot.

```text
$ git --no-pager log --all --oneline -30
2c833e6 (HEAD -> jakubs-solution, origin/jakubs-solution) fix(firmware): F3 - use MobileNetV2 [-1,1] preprocessing; add USB stream
a55638d bench: add calibration evaluation harness
697cac0 docs: add report skeleton, dataset datasheet, AI-usage statement
f8394b5 chore: add LICENSE, CITATION.cff, CONTRIBUTING.md
d6d90a9 bench: add Keras-Tuner trial distillation script
faf806f (wip/notebook-snapshot) wip: snapshot of dirty notebooks and trained .keras artifacts before consolidation
b4a11de (origin/main, origin/HEAD) edit preprocess, add camera preview python program
ea2f0a6 updated notebook and esp32 program
68ecb41 (tag: archive/pre-consolidation) Add ESP32-S3 face recognition firmware (TFLite Micro)
adeee62 Tune hyperparams, fix pipeline compat, update README
0a2c6cb (main) Add Jupyter notebook, QAT script, improve PTQ calibration
236d2ca Fix Albumentations 2.0 and MediaPipe 0.10.33 API compatibility
4a89836 Improve ML pipeline: alpha param, QAT, label smoothing, rejection, tuning
a375d14 (tag: archive/amineModel, origin/amineModel) changed the data augmentation
ec5df7e push models
19540c0 Add course project specification
caaf119 resolve augment error
4241da6 Initial commit: face recognition ML pipeline for ESP32-S3
```

The two required reachability checks from `archive/amineModel` are:

```text
$ git --no-pager log --oneline archive/amineModel..origin/main
b4a11de (origin/main, origin/HEAD) edit preprocess, add camera preview python program
ea2f0a6 updated notebook and esp32 program
ec5df7e push models
```

```text
$ git --no-pager log --oneline archive/amineModel..origin/jakubs-solution
2c833e6 (HEAD -> jakubs-solution, origin/jakubs-solution) fix(firmware): F3 - use MobileNetV2 [-1,1] preprocessing; add USB stream
a55638d bench: add calibration evaluation harness
697cac0 docs: add report skeleton, dataset datasheet, AI-usage statement
f8394b5 chore: add LICENSE, CITATION.cff, CONTRIBUTING.md
d6d90a9 bench: add Keras-Tuner trial distillation script
68ecb41 (tag: archive/pre-consolidation) Add ESP32-S3 face recognition firmware (TFLite Micro)
adeee62 Tune hyperparams, fix pipeline compat, update README
0a2c6cb (main) Add Jupyter notebook, QAT script, improve PTQ calibration
236d2ca Fix Albumentations 2.0 and MediaPipe 0.10.33 API compatibility
4a89836 Improve ML pipeline: alpha param, QAT, label smoothing, rejection, tuning
ec5df7e push models
```

Interpretation:

- `origin/main` has three commits after `archive/amineModel`.
- `origin/jakubs-solution` has eleven commits after `archive/amineModel`.
- `jakubs-solution` contains the shared `ec5df7e` model push and then the larger Jakub pipeline line.
- `main` diverges from `ec5df7e` into Rifki's notebook/preview/firmware line.
- `amineModel` diverges earlier and does not contain either branch's later work.

## 5. File-level diff summary

This section records `git --no-pager diff --stat` outputs between branch heads.
Large generated model files dominate the line counts, so the qualitative conclusion should be
based on the file list as well as the totals.

### 5.1 `origin/main..origin/jakubs-solution`

This is the required Rifki-to-Jakub comparison. It shows what the current trunk candidate has
relative to `origin/main`, and what it removes from the Rifki line.

```text
$ git --no-pager diff --stat origin/main..origin/jakubs-solution
 .gitignore                                |     10 +
 CITATION.cff                              |     23 +
 CONTRIBUTING.md                           |     40 +
 LICENSE                                   |     21 +
 README.md                                 |    208 +-
 docs/report/ai-usage.md                   |      9 +
 docs/report/dataset-datasheet.md          |     60 +
 docs/report/outline.md                    |    185 +
 esp32/dependencies.lock                   |     60 -
 esp32/main/model.c                        | 121098 +++++++++++++++++++++++++++++++------------------------------------
 esp32/main/model.h                        |     35 +-
 python/MobileNetV2_3ClassKeras.ipynb      |    255 +-
 python/MobileNetV2_3ClassKeras_Crop.ipynb |   1191 -
 python/Quantization.ipynb                 |    444 -
 python/augment.py                         |     61 +-
 python/bench/__init__.py                  |      1 +
 python/bench/build_originals_test.py      |    192 +
 python/bench/distill_tuner.py             |    423 +
 python/bench/eval_branches.py             |    234 +
 python/bench/firmware_preprocess_check.py |    171 +
 python/bench/mac_count.py                 |    226 +
 python/compare.py                         |    152 +
 python/main.py                            |    208 +-
 python/notebook.ipynb                     |    467 +
 python/preprocess.py                      |     54 +-
 python/preview_pred.py                    |    212 -
 python/qat_export.py                      |    234 +
 python/requirements.txt                   |      6 +
 python/tune.py                            |    111 +
 29 files changed, 58213 insertions(+), 68178 deletions(-)
```

Key points:

- `jakubs-solution` adds project governance/report files: `LICENSE`, `CITATION.cff`,
  `CONTRIBUTING.md`, and `docs/report/*`.
- It adds benchmark and evidence tooling under `python/bench/`.
- It keeps the modern training/export pipeline in `python/main.py`, `python/qat_export.py`,
  `python/compare.py`, and `python/tune.py`.
- It removes Rifki-only notebook and preview files from that branch comparison:
  `python/MobileNetV2_3ClassKeras_Crop.ipynb`, `python/Quantization.ipynb`, and
  `python/preview_pred.py`.
- The firmware USB-stream and preprocessing fix are already represented in
  `origin/jakubs-solution` through commit `2c833e6`, so the remaining `main` material should
  be merged only if the team wants it as documentation or archived reference.

### 5.2 Rifki-only assets relative to `jakubs-solution`

The reverse command was also checked:

```text
$ git --no-pager diff --stat origin/jakubs-solution..origin/main
29 files changed, 68178 insertions(+), 58213 deletions(-)
```

The Rifki-side assets not present in the trunk candidate are concentrated in:

- `python/MobileNetV2_3ClassKeras_Crop.ipynb`
- `python/Quantization.ipynb`
- `python/preview_pred.py`
- `esp32/dependencies.lock`
- a different generated `esp32/main/model.c` / `model.h` pair

`main` lacks the report scaffolding, benchmark harnesses, QAT export script, and tuner tooling now
present on `jakubs-solution`.

### 5.3 `origin/amineModel..origin/main`

This comparison shows how much Rifki's branch added after the stale Amine branch point.

```text
$ git --no-pager diff --stat origin/amineModel..origin/main
 README.md                                 |   157 +-
 esp32/CMakeLists.txt                      |     6 +
 esp32/dependencies.lock                   |    60 +
 esp32/main/CMakeLists.txt                 |     2 +
 esp32/main/camera.cpp                     |   117 +
 esp32/main/camera.h                       |     8 +
 esp32/main/idf_component.yml              |     6 +
 esp32/main/inference.cpp                  |   181 +
 esp32/main/inference.h                    |     7 +
 esp32/main/main.cpp                       |   201 +
 esp32/main/model.c                        | 65924 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 esp32/main/model.h                        |    15 +
 esp32/sdkconfig.defaults                  |     5 +
 python/CustomCNN_3ClassKeras.ipynb        |   641 +
 python/MobileNetV2_3ClassKeras.ipynb      |   773 +
 python/MobileNetV2_3ClassKeras_Crop.ipynb |  1191 ++
 python/Quantization.ipynb                 |   444 +
 python/augment.py                         |    66 +-
 python/preview_pred.py                    |   212 +
 19 files changed, 69901 insertions(+), 115 deletions(-)
```

This establishes that `amineModel` lacks Rifki's ESP32 firmware, notebooks, generated firmware model,
and preview tooling.

### 5.4 Amine-only assets relative to `main`

The reverse command was checked to identify Amine-only content:

```text
$ git --no-pager diff --stat origin/main..origin/amineModel
19 files changed, 115 insertions(+), 69901 deletions(-)
```

Moving from `main` back to `amineModel` mostly deletes work. The only non-deletion code item
worth reviewing from `amineModel` is the alternate `python/augment.py`; the README difference is
stale branch documentation, not a consolidation driver.

### 5.5 `origin/amineModel..origin/jakubs-solution`

This comparison shows why `jakubs-solution` is the stronger trunk candidate than `amineModel`.

```text
$ git --no-pager diff --stat origin/amineModel..origin/jakubs-solution
 .gitignore                                |    10 +
 CITATION.cff                              |    23 +
 CONTRIBUTING.md                           |    40 +
 LICENSE                                   |    21 +
 README.md                                 |    93 +-
 docs/report/ai-usage.md                   |     9 +
 docs/report/dataset-datasheet.md          |    60 +
 docs/report/outline.md                    |   185 +
 esp32/CMakeLists.txt                      |     6 +
 esp32/main/CMakeLists.txt                 |     2 +
 esp32/main/camera.cpp                     |   117 +
 esp32/main/camera.h                       |     8 +
 esp32/main/idf_component.yml              |     6 +
 esp32/main/inference.cpp                  |   181 +
 esp32/main/inference.h                    |     7 +
 esp32/main/main.cpp                       |   201 +
 esp32/main/model.c                        | 55174 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 esp32/main/model.h                        |    20 +
 esp32/sdkconfig.defaults                  |     5 +
 python/CustomCNN_3ClassKeras.ipynb        |   641 +
 python/MobileNetV2_3ClassKeras.ipynb      |   672 +
 python/augment.py                         |   111 +-
 python/bench/__init__.py                  |     1 +
 python/bench/build_originals_test.py      |   192 +
 python/bench/distill_tuner.py             |   423 +
 python/bench/eval_branches.py             |   234 +
 python/bench/firmware_preprocess_check.py |   171 +
 python/bench/mac_count.py                 |   226 +
 python/compare.py                         |   152 +
 python/main.py                            |   208 +-
 python/notebook.ipynb                     |   467 +
 python/preprocess.py                      |    54 +-
 python/qat_export.py                      |   234 +
 python/requirements.txt                   |     6 +
 python/tune.py                            |   111 +
 35 files changed, 59946 insertions(+), 125 deletions(-)
```

The result is not a close call: `jakubs-solution` contains firmware, generated INT8 deployment files,
benchmark scripts, report scaffolding, and modernized Python pipeline files absent from `amineModel`.

### 5.6 Amine-only assets relative to `jakubs-solution`

The reverse command again shows deletion of the mature trunk candidate:

```text
$ git --no-pager diff --stat origin/jakubs-solution..origin/amineModel
35 files changed, 125 insertions(+), 59946 deletions(-)
```

This supports the **ABANDON** decision for `amineModel`: reverting from `jakubs-solution` to
`amineModel` would remove firmware, generated deployment files, evaluation scripts, report
scaffolding, and the modernized training/export pipeline.

## 6. Branch-by-branch findings

### 6.1 `amineModel` / Amine

`amineModel` is stale. It is preserved by `archive/amineModel` at `a375d14`, but it should not be merged.

Evidence:

- It lacks the later ESP32 firmware tree present on both `main` and `jakubs-solution`.
- It lacks notebooks and preview tooling from `main`.
- It lacks QAT export, tuner, design-space, and benchmark/evaluation tooling from `jakubs-solution`.
- Its only plausible unique code value is an alternate `python/augment.py` implementation.
- The diff stats show that comparing either mature branch back to `amineModel` is mostly deletion.

Decision: **ABANDON**.

### 6.2 `main` / Rifki

`main` contains useful work, but it should not replace the trunk candidate wholesale.

Useful assets:

- USB-stream firmware lineage.
- `python/preview_pred.py` for camera/preview experiments.
- Crop and Quantization notebooks.
- ESP32 firmware structure and generated model artifacts.

Limitations as trunk:

- It lacks the consolidated report/documentation foundation.
- It lacks the Phase 3/4 benchmark harnesses.
- It lacks the more mature design-space/QAT/export tooling now on `jakubs-solution`.
- Several files are better treated as reference evidence than as final source of truth.

Decision: **MERGE-IN-PARTS**.

### 6.3 `jakubs-solution` / Jakub

`jakubs-solution` is the strongest trunk candidate.

Strengths:

- Design-space training pipeline in `python/main.py`.
- QAT export path in `python/qat_export.py`.
- Benchmark and calibration scripts under `python/bench/`.
- Keras-Tuner trial distillation and reported tuner results.
- Albumentations 2.0 and MediaPipe Tasks API compatibility work.
- Firmware now includes the MobileNetV2 `[-1,1]` preprocessing fix and USB-stream support.
- Report scaffolding exists under `docs/report/`.
- The deployed INT8 model has an honest originals-only accuracy of 98.33% on 60 captures.

Decision: **KEEP AS TRUNK**.

## 7. Statistical evidence summary

The current benchmark evidence comes from `bench/results/calibration_report.md`.

| Quantity | Value | Interpretation |
| --- | ---: | --- |
| Honest originals-only accuracy | 98.33% | `acc(originals_only)=0.9833`, evaluated on 60 independent original captures. |
| Honest originals-only macro-F1 | 98.33% | Macro-F1 equals 0.9833 in the report. |
| Misclassifications | 1 of 60 | Confusion matrix: one Amine image predicted as Rifki. |
| Full augmented-test accuracy | 97.69% | `acc(full_aug_test)=0.9769`, but this 780-row set is biased because it includes augmented variants of the same 60 captures. |
| Delta | -0.64 percentage points | `Δacc = acc(full_aug_test) - acc(originals_only) = -0.0064`. |
| Compute proxy | 10,695,184 MACs | Conv2D + DepthwiseConv2D + FullyConnected estimate from `bench/results/mac_count.csv`. |

The honest confusion matrix is:

```text
actual\pred  Amine  Rifki  Jakub
Amine     19      1      0
Rifki      0     20      0
Jakub      0      0     20
```

Interpretation:

- The headline result is the originals-only evaluation, not the augmented test-file count.
- The single observed error is `Amine -> Rifki`.
- The augmented-test delta is small and negative: the biased full augmented test is 0.64 percentage
  points lower than the honest originals-only test.
- This supports the existing QAT model as credible calibration evidence, while still documenting why
  Phase 2 retraining and clean validation/test separation were necessary methodology safeguards.

## 8. Consolidation rationale

The consolidation should optimize for evidence quality and maintainability rather than branch ownership.
The branch evidence supports the following rationale:

1. `jakubs-solution` has the broadest, most mature ML pipeline and report evidence.
2. `main` has useful firmware/preview work, but not enough surrounding methodology to be the trunk by itself.
3. `amineModel` does not contain enough unique functionality to justify a merge.
4. The repository now has tags preserving abandoned or pre-consolidation states, so the team can safely move
   forward without losing history.
5. The public branch history should be left intact; no force-push is needed for consolidation.

## 9. Course-rule note

Final design decision is made by the human team, not the AI tooling. This document presents evidence; the team signs off in `docs/decision.md` (Phase 5).

This audit recommends a trunk direction; the human team confirms the final design in Phase 5.

## 10. Phase 5 follow-up
Phase 5 should add the remaining archive tags, confirm whether any extra Rifki preview tooling or
Amine augmentation ideas need manual porting, and keep the final report centered on the 60 original
captures as the independent test unit.
