# Keras-Tuner trial distillation

Parsed **30 completed trials** from `python/gen/tuner/face_recognition/trial_*/trial.json`.
Scores are `val_accuracy` from the tuner run.

## Best trial

Best score: **96.79%** (trial `0025`) with `alpha=0.5, dense_units=32, dropout_1=0.2, dropout_2=0.2, learning_rate=0.0005, label_smoothing=0.1, epochs=20, bracket=1, round=1`.

## Top 10 trials

| trial | score | alpha | dense | drop1 | drop2 | lr | smooth | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0025 | 96.79% | 0.5 | 32 | 0.2 | 0.2 | 0.0005 | 0.1 | 20 |
| 0029 | 96.41% | 0.35 | 32 | 0.4 | 0.1 | 0.0005 | 0.05 | 20 |
| 0017 | 96.15% | 0.5 | 64 | 0.2 | 0.1 | 0.001 | 0 | 20 |
| 0024 | 96.15% | 0.5 | 32 | 0.2 | 0.2 | 0.0005 | 0 | 20 |
| 0016 | 96.03% | 0.35 | 128 | 0.3 | 0.2 | 0.001 | 0.15 | 20 |
| 0026 | 96.03% | 0.35 | 64 | 0.4 | 0.2 | 0.0005 | 0.05 | 20 |
| 0028 | 96.03% | 0.75 | 32 | 0.4 | 0.1 | 0.001 | 0.1 | 20 |
| 0023 | 95.64% | 0.5 | 32 | 0.2 | 0.2 | 0.0005 | 0 | 7 |
| 0015 | 95.51% | 0.35 | 128 | 0.3 | 0.2 | 0.001 | 0.15 | 7 |
| 0009 | 95.51% | 0.5 | 64 | 0.2 | 0.1 | 0.001 | 0 | 3 |

## Smallest model within 1 percentage point of best

The smallest alpha within 1pp of the best score is **alpha=0.35** (trial `0029`, 96.41%; alpha=0.35, dense_units=32, dropout_1=0.4, dropout_2=0.1, learning_rate=0.0005, label_smoothing=0.05, epochs=20, bracket=0, round=0).

## Alpha tradeoffs

| alpha | best trial | score | dense | lr | smooth | 1pp Pareto? |
| --- | --- | --- | --- | --- | --- | --- |
| 0.35 | 0029 | 96.41% | 32 | 0.0005 | 0.05 | yes |
| 0.5 | 0025 | 96.79% | 32 | 0.0005 | 0.1 | no, within 1pp dominated by smaller alpha |
| 0.75 | 0028 | 96.03% | 32 | 0.001 | 0.1 | no, within 1pp dominated by smaller alpha |

## 1pp Pareto frontier (score vs alpha)

| trial | score | alpha | dense | lr | smooth | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| 0029 | 96.41% | 0.35 | 32 | 0.0005 | 0.05 | 20 |

## Hyperparameter trends

- `alpha`: 0.35: n=10, mean=94.58%, range=91.92%–96.41%, 0.5: n=10, mean=95.58%, range=94.49%–96.79%, 0.75: n=10, mean=91.96%, range=84.74%–96.03%. Best average is `0.5`; best single trial is `0.5` (trial 0025 at 96.79%).
- `dense_units`: 32: n=13, mean=93.09%, range=84.74%–96.79%, 64: n=7, mean=95.53%, range=95.00%–96.15%, 128: n=10, mean=94.23%, range=92.18%–96.03%. Best average is `64`; best single trial is `32` (trial 0025 at 96.79%).
- `learning_rate`: 0.0001: n=7, mean=89.34%, range=84.74%–92.44%, 0.0005: n=10, mean=95.69%, range=94.87%–96.79%, 0.001: n=13, mean=95.30%, range=93.59%–96.15%. Best average is `0.0005`; best single trial is `0.0005` (trial 0025 at 96.79%).
- `label_smoothing`: 0: n=12, mean=95.05%, range=92.18%–96.15%, 0.05: n=6, mean=91.94%, range=84.74%–96.41%, 0.1: n=7, mean=94.69%, range=91.92%–96.79%, 0.15: n=5, mean=93.21%, range=86.92%–96.03%. Best average is `0`; best single trial is `0.1` (trial 0025 at 96.79%).
- `dropout_1`: 0.2: n=10, mean=93.81%, range=84.74%–96.79%, 0.3: n=9, mean=94.79%, range=92.44%–96.03%, 0.4: n=11, mean=93.64%, range=84.74%–96.41%. Best average is `0.3`; best single trial is `0.2` (trial 0025 at 96.79%).
- `dropout_2`: 0.1: n=13, mean=94.15%, range=86.92%–96.41%, 0.2: n=17, mean=93.95%, range=84.74%–96.79%. Best average is `0.1`; best single trial is `0.2` (trial 0025 at 96.79%).
- Dense-width check: dense_units=128 has a higher mean than dense_units=32 by 1.14pp. However, dense_units=32 has the better best single trial by 0.77pp. Treat this as a trend, not a dominance proof.
- Label-smoothing check: label_smoothing=0 has a higher mean than label_smoothing=0.1 by 0.36pp. However, label_smoothing=0.1 has the better best single trial by 0.64pp. Treat this as a trend, not a dominance proof.

These averages mix Hyperband brackets/epoch budgets, so they are trend indicators rather than controlled ablations.

## Recommendation for Phase 2 retraining

Use trial `0029` as the **jakubs candidate** starting point: `alpha=0.35, dense_units=32, dropout_1=0.4, dropout_2=0.1, learning_rate=0.0005, label_smoothing=0.05, epochs=20, bracket=0, round=0`. It is the smallest-alpha configuration within 1pp of the best observed score, so it preserves nearly all tuner accuracy while reducing the MobileNetV2 backbone width.

## Caveat

The tuner used the biased validation set, which is the documented leaked test set. Therefore these absolute `val_accuracy` numbers must not be quoted as final performance. Expect scores to drop by roughly 3–5 percentage points on the cleaned originals-only test set; the ranking should still be useful as design-space exploration evidence.
