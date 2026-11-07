# Paper-to-code mapping

| Paper section or experiment | Code |
|---|---|
| ROCOv2 annotations, image splits, and UMLS concept labels | `src/data.py`; input files under local `data/annotations/` and `data/images/` (not included in this repository) |
| Image preprocessing and augmentation | `src/transforms.py` |
| EfficientNet backbones and multi-label classifier | `src/model.py`; phase configurations in `configs/` |
| BCE, ASL, and hybrid loss implementations | `src/loss.py` |
| Phase A: BCE/ASL and sampling comparisons with EfficientNet-B0 | `configs/A/A1.yaml`–`configs/A/A4.yaml`; executed by `train.py` |
| Phase B: ASL hyperparameter comparisons with EfficientNet-B4 | `configs/B/B1.yaml`–`configs/B/B6.yaml`; executed by `train.py` |
| Phase C: BCE–ASL hybrid experiments with EfficientNet-B4 | `configs/C/C1.yaml`–`configs/C/C2.yaml`; executed by `train.py` |
| Training loop, optimization, checkpoint selection, and prediction export | `train.py`; shared training utilities in `src/` |
| Threshold selection and test-set evaluation | `scripts/evaluate_threshold_strategies.py`; `src/evaluator.py` |
| Paired image-level F1 significance analysis | `scripts/paired_f1_wilcoxon.py` |
| Long-tail concept-frequency figure | `scripts/make_figure1_long_tail.py` |

For the local data layout, environment setup, and runnable commands, see the [reproduction guide](reproduction.md). Dataset files and trained checkpoints are not distributed here.
