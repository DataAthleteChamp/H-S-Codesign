# Dataset Datasheet

This brief datasheet follows the spirit of *Datasheets for Datasets* (Gebru et al.) and is intended for the DTU 02214 report, not as a public data release.

## 1. Motivation

The dataset was created to satisfy the course requirement that the ML model is at least partly trained on data collected by the team. Its purpose is to train and evaluate a small embedded face-recognition system that recognizes three team members and rejects other faces.

TODO(team): add the exact capture dates, devices, and consent/ethics notes used during collection.

## 2. Composition

Classes are `Amine`, `Rifki`, and `Jakub`. Images are stored locally under `data/<class>/train/` and `data/<class>/test/`; the `data/` directory is not tracked in git because it contains personal photos.

Verified file counts on disk (post-cleanup, see § 4 / finding F1):

| Class | Original train captures | Train images on disk | Original test captures | Test images on disk | Quarantined augmented test variants |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amine | 80 | 1040 | 20 | 20 | 240 |
| Rifki | 80 | 1040 | 20 | 20 | 240 |
| Jakub | 80 | 1040 | 20 | 20 | 240 |
| **Total** | **240** | **3120** | **60** | **60** | **720** |

The training split contains 80 original captures per class. Each original has 12 derived augmentations plus the original image itself, giving `80 × 13 = 1040` train files per class. The test split holds **only the 60 original captures** (20 per class). The 720 historical augmented variants (`20 × 12 = 240` per class) were removed from `data/<class>/test/` and quarantined under `data/_quarantine/test_augmented/<class>/` with a SHA-256 manifest (`data/_quarantine/test_augmented/manifest.json`); these are retained for the augmentation-robustness diagnostic only and are never loaded by the headline evaluation.

TODO(team): confirm whether any additional non-team or rejection-class images were collected outside this three-class folder layout.

## 3. Collection Process

The images were self-captured by the team using webcam and/or phone cameras. The dataset is intended to represent the faces of the three authors only, under conditions relevant to an embedded camera demo.

TODO(team): describe the exact devices, capture locations, consent process, and whether each person captured their own images or was photographed by another team member.

## 4. Preprocessing and Labeling

Labels are derived from the folder names (`Amine`, `Rifki`, `Jakub`). The preprocessing pipeline detects faces with MediaPipe BlazeFace, crops the face with approximately 15% padding, resizes to the model input size, and normalizes pixels for the MobileNetV2 pipeline.

Augmentations are applied train-only. The historical augmented variants of the test split (`20 × 12 = 240` per class, 720 total) were removed from `data/<class>/test/` and quarantined under `data/_quarantine/test_augmented/` with a SHA-256 manifest; the cleanup procedure and code-level guards (in `python/augment.py` and `python/preprocess.py`) are described in `docs/report/methods_test_hygiene.md`. The quarantined files are still on disk and are read back by `python/bench/build_full_aug_test.py` to regenerate the n=780 augmentation-robustness diagnostic, but they never re-enter the headline test arrays.

TODO(team): record the final image size(s) used for the submitted model and whether any images failed face detection.

## 5. Conditional Factors

The dataset may include variation in lighting, head angle, distance to camera, glasses, background, and capture device, but these factors have not yet been documented in a structured metadata file.

TODO(team): describe variation captured for lighting, angle, distance, glasses, background, and camera device.

## 6. Uses

The intended use is strictly academic work within DTU 02214 — Hardware/Software Codesign. The data supports training, validation, and report figures for this course project only. It should not be redistributed publicly because it contains identifiable face images. If the optional shared SharePoint folder mentioned in `PROJECT.md` is used, access should remain limited to the team and course context.

## 7. Distribution

The dataset is not distributed with the repository and should not be uploaded to public git history. It is shared only within the project team through approved private storage.

TODO(team): document the exact private storage location and who has access.

## 8. Maintenance

The dataset will be maintained only for the course project and archived after the May 7, 2026 submission deadline. Any later community release should use either synthetic/example images or a separately consented public dataset, not these private face photos.
