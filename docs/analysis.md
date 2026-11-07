# ROCOv2 Loss Study Summary

## Objective

This study examines how different loss designs affect image-only multi-label medical concept recognition on ROCOv2.

The main questions are:

- How well can BCE-based training learn rare classes?
- What does ASL change compared with BCE?
- What happens when ASL is made more aggressive?
- Can BCE + ASL hybrids outperform pure ASL?

The goal is not to propose a new architecture, but to show clearly how the choice of loss changes model behavior.

## Problem and evaluation

ROCOv2 is a multi-label dataset, so an image can have multiple correct concepts.

Dataset summary:

- Train: `59,958`
- Validation: `9,904`
- Test: `9,927`
- Total number of concepts: `1,947`
- Average number of correct labels per image: approximately `3.36`

The official metrics are:

- `primary`: image-level mean F1
- `secondary`: an additional score on the official 14-concept subset

This study also analyzes frequency buckets:

| Bucket | Training frequency | Number of classes observed in test |
|---|---:|---:|
| `head` | `>=1000` | `18` |
| `upper-mid` | `[100, 1000)` | `219` |
| `lower-mid` | `[10, 100)` | `1210` |
| `tail` | `[2, 10)` | `303` |
| `singleton` | `1` | `0` |

These buckets are used primarily to examine whether:

- the model learns only frequent classes, or
- it also begins to recognize moderately rare classes.

The `upper-mid` bucket is the most important one for this study because:

- `head` classes are comparatively easy already;
- `tail` and `singleton` classes are very difficult; and
- `upper-mid` is the most informative range for showing whether the model is beginning to learn rare classes.

## Experimental design

The experiments are divided into three phases:

- Phase `A`: transition from BCE approaches to ASL with `EfficientNet-B0`.
- Phase `B`: ASL hyperparameter study with `EfficientNet-B4`.
- Phase `C`: `BCE + ASL` hybrids.

### Shared settings

Small-model pipeline:

- backbone: `EfficientNet-B0`
- image size: `256x256`

Large-model pipeline:

- backbone: `EfficientNet-B4`
- image size: `384x384`

All results are reported on the test set using the best global threshold selected on validation data.

---

## Phase A: EfficientNet-B0 loss/sampling progression

This phase examines how behavior changes when moving from BCE-based approaches to ASL in the same `EfficientNet-B0` pipeline.

### Methods

| Experiment | Method | Model |
|---|---|---|
| `A1` | Plain `BCE` baseline | `EfficientNet-B0` |
| `A2` | `BCE + class-balanced oversampling` | `EfficientNet-B0` |
| `A3` | Pure `ASL` (`gamma_neg=4`, `gamma_pos=0`, `clip=0.05`) | `EfficientNet-B0` |
| `A4` | `ASL + class-balanced oversampling` | `EfficientNet-B0` |

### Results

| Experiment | Threshold | Primary | Secondary | Head F1 | Upper-mid F1 | Lower-mid F1 | Tail F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A1` | `0.37` | `0.5809` | `0.9337` | `0.6464` | `0.0099` | `0.0000` | `0.0000` |
| `A2` | `0.34` | `0.5809` | `0.9280` | `0.6434` | `0.0118` | `0.0000` | `0.0000` |
| `A3` | `0.74` | `0.5882` | `0.9432` | `0.6620` | `0.0268` | `0.0038` | `0.0011` |
| `A4` | `0.79` | `0.5828` | `0.9390` | `0.6592` | `0.0211` | `0.0033` | `0.0000` |

### Findings from Phase A

- `A1` establishes a strong `head` baseline, but performance on rare classes is nearly zero.
- `A2` produces a small improvement in the `upper-mid` bucket through oversampling, while `lower-mid` and `tail` remain inactive.
- `A3` shows the first clear activation in the `upper-mid` and `lower-mid` buckets after switching to pure `ASL` in the same `EfficientNet-B0` pipeline.
- `A4` shows that adding oversampling to `ASL` does not outperform pure `ASL`.

In short:

> Oversampling on the `BCE` side does not meaningfully activate rare classes. The main change in the same B0 model comes from switching to `ASL`.

---

## Phase B: EfficientNet-B4 ASL hyperparameter experiments

This phase examines ASL with the larger `EfficientNet-B4` pipeline and studies how margin/focusing hyperparameters change the balance between the official score and rare-class activation.

### Methods

| Experiment | Method | Model |
|---|---|---|
| `B1` | `ASL` + probability margin ablation (`m=0.00`) | `EfficientNet-B4` |
| `B2` | Pure `ASL` reference (`m=0.05`) | `EfficientNet-B4` |
| `B3` | `ASL` + stronger probability margin (`m=0.10`) | `EfficientNet-B4` |
| `B4` | More aggressive `ASL` (`gamma_neg=5`) | `EfficientNet-B4` |
| `B5` | Most aggressive `ASL` (`gamma_neg=6`) | `EfficientNet-B4` |
| `B6` | `ASL` + `gamma_pos=1.0` | `EfficientNet-B4` |

### Results

| Experiment | Threshold | Primary | Secondary | Head F1 | Upper-mid F1 | Lower-mid F1 | Tail F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B1` | `0.73` | `0.5906` | `0.9475` | `0.6750` | `0.0284` | `0.0036` | `0.0000` |
| `B2` | `0.77` | `0.5916` | `0.9472` | `0.6748` | `0.0336` | `0.0042` | `0.0000` |
| `B3` | `0.79` | `0.5898` | `0.9414` | `0.6709` | `0.0405` | `0.0055` | `0.0000` |
| `B4` | `0.80` | `0.5904` | `0.9455` | `0.6719` | `0.0361` | `0.0048` | `0.0000` |
| `B5` | `0.82` | `0.5899` | `0.9438` | `0.6734` | `0.0391` | `0.0060` | `0.0000` |
| `B6` | `0.65` | `0.5878` | `0.9432` | `0.6682` | `0.0299` | `0.0037` | `0.0000` |

### Findings from Phase B

- `B2` gives the best overall result with the larger backbone.
- `B1` and `B3` form the probability-margin sweep. Although `m=0.00` and `m=0.10` are also tested, the pure reference `B2` retains the best overall balance.
- `B4` and `B5` activate rare classes further, especially in the `upper-mid` and `lower-mid` buckets.
- `B6` tests the effect of `gamma_pos` as a positive-side refinement.
- However, this rare-class activation does not consistently increase the official `primary` score.

The key observations are:

- `B2` has the best overall balance: `primary = 0.5916`.
- `B5` produces the strongest gamma-neg rare-class activation: `upper-mid F1 = 0.0391`, `lower-mid F1 = 0.0060`.
- `B3` gives the highest `upper-mid F1` in the probability-margin sweep: `0.0405`.
- However, `B3` and `B5` do not exceed `B2` on the official score.

In short:

> Compared with BCE, `ASL` clearly activates rare classes. Making ASL more aggressive can improve rare-class F1, but does not increase the image-level official F1 by the same amount.

---

## Phase C: BCE-ASL hybrid experiments

This phase tests whether controlled reintroduction of `BCE` can exceed the balance achieved by pure `ASL`.

### Methods

| Experiment | Method | Model |
|---|---|---|
| `C1` | `head-BCE / non-head-ASL` | `EfficientNet-B4` |
| `C2` | `0.95 * ASL + 0.05 * BCE` | `EfficientNet-B4` |

### Results

| Experiment | Threshold | Primary | Secondary | Head F1 | Upper-mid F1 | Lower-mid F1 |
|---|---:|---:|---:|---:|---:|---:|
| `C1` | `0.72` | `0.5798` | `0.9430` | `0.6597` | `0.0204` | `0.0020` |
| `C2` | `0.75` | `0.5879` | `0.9424` | `0.6704` | `0.0272` | `0.0019` |

### Findings from Phase C

- `C1`, which restores BCE only for head classes, is clearly worse than pure `ASL`.
- `C2` is the better hybrid, but it still falls below `B2`.

Comparison:

- `B2` (pure ASL): `primary = 0.5916`
- `C2` (best hybrid): `primary = 0.5879`

In short:

> Selectively or proportionally reintroducing `BCE` does not exceed the best balance achieved by pure `ASL`.

---

## Summary comparison

The following table presents the paper's main message at a glance:

| Experiment | Main idea | Primary | Upper-mid F1 | Brief interpretation |
|---|---|---:|---:|---|
| `A1` | Plain BCE | `0.5809` | `0.0099` | Head-dominated baseline |
| `A2` | BCE + oversampling | `0.5809` | `0.0118` | Limited effect on rare classes |
| `A3` | First pure ASL | `0.5882` | `0.0268` | First clear rare-class activation |
| `A4` | ASL + oversampling | `0.5828` | `0.0211` | Does not outperform pure ASL |
| `B2` | Larger model + pure ASL | `0.5916` | `0.0336` | Best overall model |
| `B4` | More aggressive ASL | `0.5904` | `0.0361` | Rare-class score rises; official score gains are limited |
| `B5` | Most aggressive ASL | `0.5899` | `0.0391` | Strongest rare-class result, but overall score does not rise |
| `B3` | Probability margin ablation (`m=0.10`) | `0.5898` | `0.0405` | Further rare-class activation, but still below the overall score |
| `C2` | Best hybrid | `0.5879` | `0.0272` | Below pure ASL |

---

## All experiments

This table collects all active experiments reported in the paper and supports direct comparison across experiments.

| Experiment | Threshold | Primary | Secondary | Head F1 | Upper-mid F1 | Lower-mid F1 | Tail F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A1` | `0.37` | `0.5809` | `0.9337` | `0.6464` | `0.0099` | `0.0000` | `0.0000` |
| `A2` | `0.34` | `0.5809` | `0.9280` | `0.6434` | `0.0118` | `0.0000` | `0.0000` |
| `A3` | `0.74` | `0.5882` | `0.9432` | `0.6620` | `0.0268` | `0.0038` | `0.0011` |
| `A4` | `0.79` | `0.5828` | `0.9390` | `0.6592` | `0.0211` | `0.0033` | `0.0000` |
| `B1` | `0.73` | `0.5906` | `0.9475` | `0.6750` | `0.0284` | `0.0036` | `0.0000` |
| `B2` | `0.77` | `0.5916` | `0.9472` | `0.6748` | `0.0336` | `0.0042` | `0.0000` |
| `B3` | `0.79` | `0.5898` | `0.9414` | `0.6709` | `0.0405` | `0.0055` | `0.0000` |
| `B4` | `0.80` | `0.5904` | `0.9455` | `0.6719` | `0.0361` | `0.0048` | `0.0000` |
| `B5` | `0.82` | `0.5899` | `0.9438` | `0.6734` | `0.0391` | `0.0060` | `0.0000` |
| `B6` | `0.65` | `0.5878` | `0.9432` | `0.6682` | `0.0299` | `0.0037` | `0.0000` |
| `C1` | `0.72` | `0.5798` | `0.9430` | `0.6597` | `0.0204` | `0.0020` | `0.0000` |
| `C2` | `0.75` | `0.5879` | `0.9424` | `0.6704` | `0.0272` | `0.0019` | `0.0000` |

Several main patterns are clear in this table:

- Phase `A` shows the transition from BCE approaches to `ASL` with the B0 pipeline; the main improvement comes with `A3`.
- Phase `B` shows how pure ASL hyperparameters affect the balance between the official score and rare-class activation in the B4 pipeline.
- `B2` gives the best overall score.
- `B1` and `B3` show the effect of the probability-margin value, but the margin sweep does not exceed the overall balance of `B2`.
- `B4` and `B5` activate rare classes further, but this does not raise the official score by the same amount.
- Although `B6` adds positive-side ASL refinement, it does not exceed the balance achieved by pure `B2`.
- The hybrids in Phase `C` do not outperform the pure ASL reference `B2`.

---

## Conclusion

The clearest findings of this study are:

1. BCE-based methods are insufficient for activating rare classes on ROCOv2.
2. ASL produces a clear improvement in activation, particularly for `upper-mid` classes.
3. More aggressive ASL settings improve rare-class performance but do not improve the official image-level F1 by the same amount.
4. `BCE + ASL` hybrids do not outperform pure `ASL`.

The main academic contribution of this project can therefore be summarized as follows:

> On ROCOv2, a controlled comparison of the `BCE` and `ASL` loss functions shows a clear difference in rare-class behavior. `ASL` meaningfully activates `upper-mid` classes, while BCE-based approaches do not produce the same effect.
