# Methods note: test-split hygiene (F1 fix)

> Goes into the report's *Evaluation methodology* section. Length: ≤ 200
> words on purpose, so it can drop straight into the report without
> editing.

We treat data augmentation as a strictly training-time regulariser,
following the canonical computer-vision pipelines (Krizhevsky et al.
2012, He et al. 2016, Szegedy et al. 2016) and the textbook
formulation in Goodfellow, Bengio and Courville (*Deep Learning*,
2016, §5.3 and §7.4). Augmentation is therefore applied only in the
`train` split. Augmented variants are never persisted under
`data/<person>/test/`; the test arrays contain exactly one row per
original capture, and the unit of evaluation is the capture, not the
file. This rules out the train–test leakage modes catalogued in
Kapoor & Narayanan (*Patterns* 4(9), 2023, arXiv:2207.07048) and the
duplicate-test-row failure mode in Sculley et al. (NeurIPS 2015) and
Roelofs et al. (NeurIPS 2019). The historical contamination (finding
F1) was remediated by quarantining 720 augmented files with a
SHA-256 manifest, and the policy is now enforced in code:
`python/augment.py` raises if any augmented file appears under
`data/<person>/test/`, and `python/preprocess.py` skips augmentation
suffixes and asserts balanced per-class test counts. The
augmentation-robustness panel (n=780) is regenerated from the
quarantine by `python/bench/build_full_aug_test.py`, never by
re-polluting `test/`, and is reported as a non-independent diagnostic
only.

## References

- Goodfellow, Bengio, Courville (2016). *Deep Learning.* MIT Press.
  §5.3 (Hyperparameters and validation), §7.4 (Dataset augmentation).
- Kapoor S., Narayanan A. (2023). "Leakage and the Reproducibility
  Crisis in ML-based Science." *Patterns* 4(9), 100804.
  arXiv:2207.07048.
- Sculley D. et al. (2015). "Hidden Technical Debt in Machine Learning
  Systems." *NeurIPS 2015*.
- Roelofs R. et al. (2019). "A Meta-Analysis of Overfitting in
  Machine Learning." *NeurIPS 2019*.
- Krizhevsky A., Sutskever I., Hinton G. (2012). "ImageNet
  Classification with Deep Convolutional Neural Networks." *NeurIPS
  2012*.
- He K., Zhang X., Ren S., Sun J. (2016). "Deep Residual Learning for
  Image Recognition." *CVPR 2016*.
- Szegedy C. et al. (2016). "Rethinking the Inception Architecture for
  Computer Vision." *CVPR 2016*.
