# Report Outline

Target length: 18–25 pages, excluding source-code listings and large appendices. Keep the main text aligned with the DTU 02214 report requirements and move exhaustive tables to appendices.

## 1. Application Description

Purpose: Explain the face-recognition application, why it is suitable for the XIAO ESP32-S3 Sense, and what the system is expected to achieve. This section should make the project scope and success criteria explicit before presenting implementation details.

Required content:
- Purpose: recognize Amine, Rifki, and Jakub while rejecting non-team faces.
- Scope: camera-based embedded ML demo for DTU 02214, with no hardware access for final validation if that remains true.
- Requirements: accuracy, rejection behavior, latency, flash size, RAM/tensor-arena, and usability targets.
- How it works: camera capture → face crop → resize/normalize → MobileNetV2 classifier → confidence threshold → display/log result.

Figures/tables:
- System overview diagram, e.g. `docs/architecture.svg`.
- Requirement target vs achieved table sourced from `bench/results/*.csv`.
- Example prediction or webcam-proxy screenshots saved as `bench/results/*.png` if available.

## 2. Dataset Description

Purpose: Describe the self-collected dataset and its limitations so the evaluation claims use the correct unit of analysis. This section should point readers to the dataset datasheet for fuller ethical and maintenance notes.

Required content:
- Classes: Amine, Rifki, Jakub.
- Conditional factors: lighting, angle, distance, glasses, background, and capture device variation.
- Amount: original train/test capture counts, augmented-on-disk counts, and effective independent test `n=60`.
- Sources and format: team-collected images under `data/<class>/<split>/`, not committed to git.
- Link to `docs/report/dataset-datasheet.md`.

Figures/tables:
- Dataset count table, generated or checked against `bench/results/*.csv`.
- Example anonymized/cropped samples if permitted, saved as `bench/results/*.png`.
- Split diagram showing originals vs train-only augmentations.

## 3. Design and Implementation

Purpose: Present the end-to-end hardware/software design and the explored design space. The section should connect course concepts to concrete trade-offs in model architecture, quantization, firmware constraints, and evaluation evidence.

### 3.1 Training, Conversion, and Optimization Pipeline Overview

Purpose: Summarize the Python pipeline from raw images to deployable INT8 artifacts. Emphasize reproducibility boundaries and the corrected train/validation/test separation.

Required content:
- Augmentation, MediaPipe BlazeFace crop with 15% padding, resize, and normalization.
- MobileNetV2 transfer learning, fine-tuning, label smoothing/dropout, and rejection threshold.
- QAT/PTQ conversion to TFLite INT8 and generation of ESP32 `model.c` / `model.h`.
- Multi-seed reporting and stored benchmark artifacts.

Figures/tables:
- Pipeline diagram with artifact names.
- Training curves and conversion summary from `bench/results/*.png` and `bench/results/*.csv`.
- Model-size and quantization comparison table from `bench/results/*.csv`.

### 3.2 ESP32 Application Code Overview

Purpose: Explain how the embedded C/C++ application maps the model pipeline onto the XIAO ESP32-S3 Sense. Make memory, camera, and preprocessing constraints understandable without listing full source code.

Required content:
- Camera configuration and frame capture path.
- Face crop/resize/preprocess path and tensor quantization.
- TFLite Micro interpreter setup, tensor arena, model loading, and output handling.
- Logging or USB-streaming behavior used for debugging and desktop proxy tests.

Figures/tables:
- Firmware module diagram for `esp32/main/*.cpp`.
- Memory/flash/tensor-arena table from `bench/results/*.csv`.
- Representative serial output or proxy screenshot as `bench/results/*.png`.

### 3.3 Design Space and Parameters

Purpose: Define the parameters varied during exploration and why each parameter matters on the ESP32-S3. Separate predeclared parameters from opportunistic debugging changes.

Required content:
- MobileNetV2 alpha/depth multiplier, image size, dense head width, dropout, label smoothing, fine-tune depth.
- Quantization choice: QAT vs PTQ.
- Rejection threshold sweep and class-confidence behavior.
- Resource proxies: TFLite size, MAC count, and tensor arena estimate.

Figures/tables:
- Full design-space table from `docs/design-space.md` and `bench/results/*.csv`.
- Pareto plots `bench/results/pareto_size.png` and `bench/results/pareto_macs.png`.
- Per-trial tuner summary from `bench/results/*.csv`.

### 3.4 Justification of Design Choices and Trade-offs

Purpose: Justify the final implementation using evidence rather than preference. Tie the final branch/model choice to accuracy, uncertainty, memory footprint, and course theory.

Required content:
- Lexicographic decision rule: macro-F1, exact McNemar, then size/MAC tie-breakers.
- Trade-off discussion for transfer learning, quantization, input resolution, and rejection thresholds.
- Human team sign-off in `docs/decision.md`; AI provided evidence, not the final decision.
- Limitations from no on-device hardware validation.

Figures/tables:
- Decision summary table from `bench/results/*.csv`.
- Pareto frontier plots from `bench/results/*.png`.
- Final chosen configuration table with links to generated artifacts.

## 4. Verification

Purpose: Demonstrate that the application meets its stated requirements as far as the available test setup permits. This section must be explicit about statistical methodology, independent test units, and proxy limitations.

### 4.1 Model Performance Evaluation

Purpose: Report classifier performance on the originals-only test set with statistical methods that match the small, paired sample size. Avoid treating augmented variants as independent evidence.

Required content:
- Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, and rejection-threshold behavior.
- Originals-only test set: 20 original captures per class, effective independent `n=60`.
- Exact McNemar test for paired model comparisons, not only asymptotic chi-square.
- Cluster bootstrap confidence intervals by original capture.

Figures/tables:
- Confusion matrices and calibration/rejection plots from `bench/results/*.png`.
- Metrics, exact-McNemar, paired-permutation, and bootstrap tables from `bench/results/*.csv`.
- Prediction/probability artifacts from `bench/results/*.npz` summarized into tables.

### 4.2 Real-world Test Results

Purpose: Present the desktop-webcam proxy test used because the XIAO ESP32-S3 Sense hardware was not available for final validation. Clearly label the result as a proxy rather than an on-device measurement.

Required content:
- Test protocol: live frames for team members and non-team faces if available.
- Environment: webcam, lighting, distance, and preprocessing path.
- Metrics: accepted accuracy, false accept/reject examples, and qualitative failure cases.
- Difference between desktop proxy latency and expected embedded latency.

Figures/tables:
- Webcam-proxy results table from `bench/results/*.csv`.
- Example accepted/rejected frames or logs in `bench/results/*.png`.
- Latency/resource proxy table from `bench/results/*.csv`.

### 4.3 Found-and-fixed Firmware Bug

Purpose: Document the preprocessing mismatch found during review and how it was corrected. This strengthens reproducibility by showing that the deployed input distribution matches the training pipeline.

Required content:
- Bug: firmware normalized camera bytes to `[0,1]` while training used MobileNetV2 `[-1,1]` preprocessing.
- Impact: INT8 input used only the non-negative half of the expected range, likely reducing deployment accuracy.
- Fix: use `(pixel / 127.5) - 1.0` before quantization and verify against a desktop reference vector.
- Regression check and remaining limitations.

Figures/tables:
- Before/after preprocessing range table from `bench/results/*.csv`.
- Reference-vector comparison or histogram from `bench/results/*.png`.

## 5. Team Member Contributions

Purpose: Account for each team member's contribution to the project and report. Keep this specific enough for assessment and signed off by the team.

Required content:
- Amine: TODO(team) contribution summary.
- Rifki: TODO(team) contribution summary.
- Jakub: TODO(team) contribution summary.
- Shared review, testing, and report-writing responsibilities.

Figures/tables:
- Contribution matrix or short table, optionally sourced from issue/PR history.

## 6. Use of AI

Purpose: Disclose AI assistance candidly while preserving the course rule that AI must not make design decisions. The final paragraph should be adapted from `docs/report/ai-usage.md`.

Required content:
- Where AI was used: planning, review, methodology suggestions, harness/doc generation.
- Where AI was not used: final design decisions, winning-branch selection, dataset collection/labeling, course-theory interpretation.
- Human verification and lack of direct AI write access to `main`.

Figures/tables:
- No required figure; include a short disclosure box or appendix link.

## 7. Appendices

Purpose: Move supporting evidence out of the main narrative while keeping the report auditable. Appendices should be referenced from the main text, not used as a dumping ground.

Required content:
- Branch audit from `docs/branch-audit.md`.
- Design-space tables from `docs/design-space.md` and `bench/results/*.csv`.
- Full per-trial tuner table from `bench/results/*.csv`.
- Additional exact-McNemar, bootstrap, rejection-sweep, and MAC-count outputs.

Figures/tables:
- Full CSV-derived tables from `bench/results/*.csv`.
- Supplementary Pareto, confusion matrix, and tuning plots from `bench/results/*.png`.
