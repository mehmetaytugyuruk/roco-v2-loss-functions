# Reproduction guide

The study uses three experiment phases: loss and sampling comparisons with EfficientNet-B0, Asymmetric Loss settings with EfficientNet-B4, and BCE–ASL hybrid losses with EfficientNet-B4. The repository's implementation paths are summarized in [Paper-to-code mapping](paper-to-code-mapping.md), and the reported results are summarized in [analysis.md](analysis.md).

## Dataset

The experiments use ROCOv2, a radiology image dataset with captions and UMLS concept annotations. The dataset paper describes its source and data records: [ROCOv2 dataset paper](https://doi.org/10.1038/s41597-024-03496-6). Obtain the dataset from its [official Zenodo record](https://zenodo.org/records/10821435) and follow the dataset authors' terms and attribution requirements.

This repository does not redistribute ROCOv2 images or annotations. The dataset includes source-specific license information; check the relevant terms before using or sharing any source image. Keep the downloaded files in the layout expected by the configurations:

```text
data/
├── annotations/
│   ├── train_concepts.csv
│   ├── valid_concepts.csv
│   ├── test_concepts.csv
│   ├── train_concepts_manual.csv
│   ├── valid_concepts_manual.csv
│   ├── test_concepts_manual.csv
│   └── cui_mapping.csv
└── images/
    ├── train/
    ├── valid/
    └── test/
```

Threshold-analysis scripts use both the concept annotations and the manually curated concept annotations. The training script checks split integrity before starting.

## Environment setup

Use Python 3 and install a PyTorch/torchvision build compatible with your operating system and accelerator. `requirements.txt` lists the required packages, but the original experiment environment was not captured as a pinned lock file, so exact package-version reproduction is not guaranteed by that file alone.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. For GPU training, install the PyTorch build that matches your CUDA setup before running the scripts.

## Run an experiment

From the repository root, train the EfficientNet-B4 ASL configuration used as the best overall reference in the paper:

```bash
python train.py --config configs/B/B2.yaml --seed 42 --num-workers 2
```

Pass `--output-dir` to use a different output location. Training writes checkpoints, a training log, and validation/test probability CSVs under the run directory. These generated files can be large and are excluded by `.gitignore`.

The experiment configurations are grouped by phase:

- `configs/A/`: EfficientNet-B0 BCE, sampling, and ASL comparisons.
- `configs/B/`: EfficientNet-B4 ASL hyperparameter experiments.
- `configs/C/`: BCE–ASL hybrid experiments.

## Evaluation and analysis

After training, select thresholds using the validation predictions and evaluate the test predictions:

```bash
python scripts/evaluate_threshold_strategies.py --experiments runs/B/B2_42
```

This writes `results.json` in the run directory. Run the paired image-level F1 Wilcoxon analysis using its default comparisons with:

```bash
python scripts/paired_f1_wilcoxon.py
```

Regenerate the long-tail concept-frequency figure when the training concept annotations and CUI mapping are present:

```bash
python scripts/make_figure1_long_tail.py
```

The script writes `docs/figures/figure1.pdf` from the local ROCOv2 annotations; the dataset itself is not included here.

## License and third-party materials

The MIT license in this repository applies to the project's source code and experiment configurations only. It does not grant rights to the Springer chapter, its publication figure, ROCOv2 data, source images, or materials owned by third parties. See [`LICENSE`](../LICENSE) and the dataset's own documentation for their respective terms.
