from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_PDF = OUTPUT_DIR / "figure1.pdf"

SPLITS = ("train",)
TOP_K_PER_BUCKET = 6

MANUAL_TARGETS = {
    "Head": [1200, 2200, 4100, 8300, 16600, 20000],
    "Mid":  [115,  180,  282,  440,  688,   986],
    "Tail": [1,    2,    4,    9,    20,    45],
}

HEAD_COLOR = "#2459A6"
MID_COLOR  = "#4B91D1"
TAIL_COLOR = "#D78A4B"
TEXT_COLOR = "#1E2A36"
GRID_COLOR = "#D9E2EC"
PURE_CUI_RE = re.compile(r"^C\d+$")


def load_counts_and_names() -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    for split in SPLITS:
        with (ANNOTATIONS_DIR / f"{split}_concepts.csv").open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cuis = [c.strip() for c in (row.get("CUIs") or "").split(";") if c.strip()]
                counts.update(cuis)

    names: dict[str, str] = {}
    with (ANNOTATIONS_DIR / "cui_mapping.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names[row["CUI"]] = row["Canonical name"]
    return counts, names


def prepare_rows(counts: Counter[str], names: dict[str, str]):
    bucket_defs = [
        ("Head", lambda v: v >= 1000,       HEAD_COLOR),
        ("Mid",  lambda v: 100 <= v < 1000, MID_COLOR),
        ("Tail", lambda v: v < 100,         TAIL_COLOR),
    ]

    rows = []
    bucket_meta = []
    start = 0
    for bucket_name, predicate, color in bucket_defs:
        items = [
            (names.get(cui, cui), freq, bucket_name, color)
            for cui, freq in counts.items()
            if predicate(freq)
        ]
        items.sort(key=lambda x: (-x[1], x[0]))

        readable = [item for item in items if not PURE_CUI_RE.match(item[0])]
        if len(readable) >= TOP_K_PER_BUCKET:
            items = readable
        targets = MANUAL_TARGETS.get(bucket_name)
        items = select_representative_items(items, TOP_K_PER_BUCKET, targets=targets)

        # Because the chart places later rows lower on the page, descending order here
        # makes each bucket increase from top to bottom in the final figure.
        items.sort(key=lambda x: x[1], reverse=True)
        rows.extend(items)
        bucket_meta.append(
            {
                "name": bucket_name,
                "color": color,
                "count_total": sum(1 for _cui, freq in counts.items() if predicate(freq)),
                "start": start,
                "end": start + len(items) - 1,
            }
        )
        start += len(items)

    return rows, bucket_meta


def select_representative_items(items, k, targets=None):
    if len(items) <= k:
        return items

    by_freq = {}
    for item in items:
        by_freq.setdefault(item[1], []).append(item)

    unique_freqs = sorted(by_freq.keys())
    if targets is not None:
        target_freqs = [float(t) for t in targets]
    elif len(unique_freqs) >= k:
        log_targets = np.logspace(
            np.log10(max(unique_freqs[0], 1)),
            np.log10(max(unique_freqs[-1], 1)),
            num=k,
        )
        target_freqs = log_targets.tolist()
    else:
        target_freqs = unique_freqs.copy()

    chosen_freqs = []
    remaining = unique_freqs.copy()
    for target in target_freqs:
        if not remaining:
            break
        best = min(remaining, key=lambda freq: (abs(np.log(freq) - np.log(target)), freq))
        chosen_freqs.append(best)
        remaining.remove(best)

    if len(chosen_freqs) < k:
        for freq in unique_freqs:
            if freq not in chosen_freqs:
                chosen_freqs.append(freq)
            if len(chosen_freqs) == k:
                break

    chosen_freqs = sorted(chosen_freqs[:k])

    selected = []
    used_labels = set()
    for freq in chosen_freqs:
        candidates = sorted(by_freq[freq], key=lambda x: (len(x[0]), x[0]))
        for candidate in candidates:
            if candidate[0] not in used_labels:
                selected.append(candidate)
                used_labels.add(candidate[0])
                break

    if len(selected) < k:
        for item in sorted(items, key=lambda x: (x[1], len(x[0]), x[0])):
            if item[0] not in used_labels:
                selected.append(item)
                used_labels.add(item[0])
            if len(selected) == k:
                break

    return selected[:k]


def draw(rows, bucket_meta) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
        }
    )

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(14.0, 7.4), constrained_layout=True)
    bar_left = 0.8
    bar_widths = [v - bar_left for v in values]
    bars = ax.barh(
        y,
        bar_widths,
        left=bar_left,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        height=0.9,
    )

    ax.set_yticks(y, labels)
    ax.margins(y=0.01)
    ax.set_xlabel("Training-set frequency (log scale)")

    xmax = max(values) * 1.35
    ax.set_xlim(0.8, xmax)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    # Bucket separators and direct labels.
    for meta in bucket_meta[:-1]:
        ax.axhline(meta["end"] + 0.5, color="#B8C2CC", linewidth=1.1, linestyle=(0, (4, 4)))
    for meta in bucket_meta:
        midpoint = (meta["start"] + meta["end"]) / 2
        ax.text(
            xmax * 0.68,
            midpoint,
            f"{meta['name']} (n={meta['count_total']})",
            color=meta["color"],
            fontsize=11.2,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.9},
        )

    for bar, value in zip(bars, values):
        value_x = value * 1.08 if value > 1 else 1.08
        ax.text(
            value_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=10.5,
            color=TEXT_COLOR,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    counts, names = load_counts_and_names()
    rows, bucket_meta = prepare_rows(counts, names)
    draw(rows, bucket_meta)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
