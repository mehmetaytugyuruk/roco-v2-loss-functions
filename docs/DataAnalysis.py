import csv
import json
import os
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_ROOT_DIR = DATA_DIR / "images"
SAVE_DIR = PROJECT_ROOT / "docs"
OUTPUT_JSONL = SAVE_DIR / "DataAnalysis.jsonl"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

NUM_WORKERS = 16
COOC_MIN_FREQ = 50


def summarize(arr):
    if len(arr) == 0:
        return {}
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def read_concepts_csv(path):
    image_cui_map = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV header missing: {path}")

        has_id = "ID" in reader.fieldnames
        has_cuis = "CUIs" in reader.fieldnames
        if not (has_id and has_cuis):
            raise ValueError(
                f"Expected columns ['ID', 'CUIs'] but got {reader.fieldnames} in {path}"
            )

        for row in reader:
            img_id = (row.get("ID") or "").strip()
            cui_raw = (row.get("CUIs") or "").strip()
            if not img_id:
                continue
            cuis = [c.strip() for c in cui_raw.split(";") if c.strip()]
            image_cui_map[img_id] = cuis

    return image_cui_map


def load_all_concepts():
    image_cui_map = {}
    all_cuis = []
    cardinalities = []
    split_counts = {}

    for split in SPLITS:
        csv_path = ANNOTATIONS_DIR / f"{split}_concepts.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing concepts file: {csv_path}")

        split_map = read_concepts_csv(csv_path)
        split_counts[split] = len(split_map)

        overlap = set(image_cui_map).intersection(split_map)
        if overlap:
            sample = sorted(list(overlap))[:3]
            raise ValueError(
                f"Duplicate IDs across splits in concepts CSVs. Example IDs: {sample}"
            )

        image_cui_map.update(split_map)
        for cuis in split_map.values():
            all_cuis.extend(cuis)
            cardinalities.append(len(cuis))

    return image_cui_map, all_cuis, np.asarray(cardinalities), Counter(cardinalities), split_counts


def scan_images():
    image_index = {}
    split_counts = {}
    duplicate_ids = []

    for split in SPLITS:
        split_dir = IMAGES_ROOT_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {split_dir}")

        split_count = 0
        for entry in os.scandir(split_dir):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_id = Path(entry.name).stem
            if img_id in image_index:
                duplicate_ids.append(img_id)
                continue

            image_index[img_id] = {
                "name": entry.name,
                "split": split,
                "path": entry.path,
            }
            split_count += 1

        split_counts[split] = split_count

    return image_index, split_counts, duplicate_ids


def analyze_image(img_id, meta):
    out = {
        "id": img_id,
        "name": meta["name"],
        "split": meta["split"],
        "w": 0,
        "h": 0,
        "mode": "ERROR",
        "error": None,
    }
    try:
        with Image.open(meta["path"]) as img:
            out["w"], out["h"] = img.size
            out["mode"] = img.mode
    except Exception as exc:
        out["error"] = str(exc)

    return out


def analyze_top_cui_overlap(image_cui_map, top_n=10, top_cuis_list=None):
    if not top_cuis_list:
        return {}

    top_set = [c["cui"] for c in top_cuis_list[:top_n]]
    counts = {c: 0 for c in top_set}
    pair_counts = defaultdict(int)
    total = len(image_cui_map)

    for cuis in image_cui_map.values():
        cui_set = set(cuis)
        for c in top_set:
            if c in cui_set:
                counts[c] += 1
        present = [c for c in top_set if c in cui_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_counts[(present[i], present[j])] += 1

    overlap = []
    for (c1, c2), cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        min_freq = min(counts[c1], counts[c2])
        overlap.append(
            {
                "pair": [c1, c2],
                "co_count": cnt,
                "overlap_ratio_of_rarer": round(cnt / min_freq, 4) if min_freq else 0,
                "pct_of_total": round(cnt / total * 100, 2) if total else 0,
            }
        )
    return overlap


def analyze_resolution_vs_cui_frequency(img_results, image_cui_map, cui_counts):
    tail_cuis = {c for c, f in cui_counts.items() if f <= 9}
    head_cuis = {c for c, f in cui_counts.items() if f >= 1000}

    tail_images, head_images, other_images = [], [], []
    for result in img_results:
        if result["error"]:
            continue
        cuis = set(image_cui_map.get(result["id"], []))
        has_tail = bool(cuis & tail_cuis)
        has_head = bool(cuis & head_cuis)
        if has_tail:
            tail_images.append(result)
        elif has_head:
            head_images.append(result)
        else:
            other_images.append(result)

    def profile(items):
        if not items:
            return {}
        return {
            "count": len(items),
            "avg_area": round(float(np.mean([r["w"] * r["h"] for r in items])), 1),
        }

    return {
        "tail_cui_images": profile(tail_images),
        "head_cui_images": profile(head_images),
        "other_images": profile(other_images),
    }


def main():
    sep = "=" * 65
    print(sep)
    print("  ROCOv2 Data Analysis")
    print(sep)

    print("\n[1/8] Reading concepts CSV files...")
    image_cui_map, all_cuis, cardinalities, card_counter, concept_split_counts = load_all_concepts()

    cui_counts = Counter(all_cuis)
    print(f"    Total samples   : {len(image_cui_map):,}")
    print(f"    Unique CUI      : {len(cui_counts):,}")
    print(f"    Total labels    : {len(all_cuis):,}")
    for split, count in concept_split_counts.items():
        print(f"    {split:<6} concepts : {count:,}")

    print("\n[2/8] CUI frequency and long-tail analysis...")
    freq_vals = np.asarray(list(cui_counts.values()))
    sorted_f = np.sort(freq_vals)
    n = len(sorted_f)
    if n == 0 or sorted_f.sum() == 0:
        gini = 0.0
    else:
        gini = float(
            (2 * np.sum(np.arange(1, n + 1) * sorted_f) - (n + 1) * sorted_f.sum())
            / (n * sorted_f.sum())
        )

    label_dist = {
        "head_gte_1000": int((freq_vals >= 1000).sum()) if len(freq_vals) else 0,
        "mid_100_to_999": int(((freq_vals >= 100) & (freq_vals < 1000)).sum()) if len(freq_vals) else 0,
        "mid_10_to_99": int(((freq_vals >= 10) & (freq_vals < 100)).sum()) if len(freq_vals) else 0,
        "tail_2_to_9": int(((freq_vals >= 2) & (freq_vals < 10)).sum()) if len(freq_vals) else 0,
        "singleton": int((freq_vals == 1).sum()) if len(freq_vals) else 0,
    }
    print(f"    Gini coefficient: {gini:.4f}")

    print("\n[3/8] Computing per-image entropy...")
    total_labels = len(all_cuis)
    p_cui = {c: v / total_labels for c, v in cui_counts.items()} if total_labels else {}
    per_img_ent = []
    for cuis in image_cui_map.values():
        if not cuis:
            per_img_ent.append(0.0)
            continue
        probs = np.asarray([p_cui.get(c, 1e-12) for c in cuis])
        probs /= probs.sum()
        per_img_ent.append(float(-np.sum(probs * np.log2(probs + 1e-12))))
    per_img_ent = np.asarray(per_img_ent)

    print(f"\n[4/8] Computing co-occurrence (min_freq={COOC_MIN_FREQ})...")
    co_counts = Counter()
    cui_img_cnt = defaultdict(int)
    for cuis in image_cui_map.values():
        unique_cuis = sorted(set(cuis))
        for cui in unique_cuis:
            cui_img_cnt[cui] += 1
        for i in range(len(unique_cuis)):
            for j in range(i + 1, len(unique_cuis)):
                co_counts[(unique_cuis[i], unique_cuis[j])] += 1

    co_list = []
    for (c1, c2), cnt in co_counts.most_common(5000):
        if cnt < COOC_MIN_FREQ:
            continue
        co_list.append(
            {
                "pair": [c1, c2],
                "freq": cnt,
                "c1_image_freq": int(cui_img_cnt[c1]),
                "c2_image_freq": int(cui_img_cnt[c2]),
            }
        )
    co_by_freq = sorted(co_list, key=lambda x: -x["freq"])
    print(f"    Pairs after filter: {len(co_list):,}")

    print("\n[5/8] Scanning image files...")
    image_index, image_split_counts, duplicate_image_ids = scan_images()
    image_ids = sorted(image_index.keys())
    print(f"    Total images on disk: {len(image_ids):,}")
    for split, count in image_split_counts.items():
        print(f"    {split:<6} images   : {count:,}")
    if duplicate_image_ids:
        print(f"    Duplicate image IDs : {len(duplicate_image_ids):,}")

    csv_ids = set(image_cui_map.keys())
    file_ids = set(image_ids)
    in_csv_not_disk = csv_ids - file_ids
    in_disk_not_csv = file_ids - csv_ids

    print(f"\n[6/8] Extracting image dimension stats ({NUM_WORKERS} workers)...")
    img_results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(analyze_image, image_id, image_index[image_id]): image_id
            for image_id in image_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), ncols=70):
            img_results.append(future.result())

    ok_results = [r for r in img_results if not r["error"]]
    err_results = [r for r in img_results if r["error"]]

    widths = np.asarray([r["w"] for r in ok_results])
    heights = np.asarray([r["h"] for r in ok_results])
    areas = widths * heights
    aspect_ratios = widths / np.where(heights == 0, 1, heights)
    mode_counter = Counter(r["mode"] for r in ok_results)

    print("\n[7/8] Aspect ratio and relation analysis...")
    ar_buckets = Counter()
    for ar in aspect_ratios:
        if ar < 0.60:
            ar_buckets["portrait_tall"] += 1
        elif ar < 0.90:
            ar_buckets["portrait"] += 1
        elif ar < 1.10:
            ar_buckets["square"] += 1
        elif ar < 1.60:
            ar_buckets["landscape"] += 1
        else:
            ar_buckets["landscape_wide"] += 1

    top_100_cuis = [
        {"cui": c, "freq": f, "coverage_pct": round(f / len(image_cui_map) * 100, 2)}
        for c, f in cui_counts.most_common(100)
    ]
    top_cui_overlap = analyze_top_cui_overlap(
        image_cui_map, top_n=10, top_cuis_list=top_100_cuis
    )
    resolution_vs_freq = analyze_resolution_vs_cui_frequency(
        img_results, image_cui_map, cui_counts
    )

    low_entropy_pct = (
        round(float((per_img_ent < np.percentile(per_img_ent, 10)).mean() * 100), 2)
        if len(per_img_ent)
        else 0.0
    )
    high_entropy_pct = (
        round(float((per_img_ent > np.percentile(per_img_ent, 90)).mean() * 100), 2)
        if len(per_img_ent)
        else 0.0
    )
    avg_labels_per_image = round(float(cardinalities.mean()), 4) if len(cardinalities) else 0.0

    stats = {
        "overview": {
            "total_samples_csv": len(image_cui_map),
            "total_image_files": len(image_ids),
            "concept_samples_by_split": concept_split_counts,
            "image_files_by_split": image_split_counts,
            "duplicate_image_ids_count": len(duplicate_image_ids),
            "duplicate_image_ids_examples": sorted(duplicate_image_ids)[:20],
            "corrupt_or_unreadable": len(err_results),
            "in_csv_not_on_disk": len(in_csv_not_disk),
            "on_disk_not_in_csv": len(in_disk_not_csv),
            "in_csv_not_on_disk_ids": sorted(list(in_csv_not_disk))[:20],
            "on_disk_not_in_csv_ids": sorted(list(in_disk_not_csv))[:20],
            "error_examples": [r["name"] for r in err_results[:10]],
        },
        "image_dimensions": {
            "widths": summarize(widths),
            "heights": summarize(heights),
            "areas": summarize(areas),
            "aspect_ratios": summarize(aspect_ratios),
            "aspect_ratio_buckets": dict(ar_buckets),
            "color_modes": dict(mode_counter),
        },
        "label_statistics": {
            "unique_cuis": len(cui_counts),
            "total_label_assignments": int(len(all_cuis)),
            "avg_labels_per_image": avg_labels_per_image,
            "gini_coefficient": round(gini, 4),
            "cardinality": {
                "stats": summarize(cardinalities),
                "histogram": {int(k): int(v) for k, v in sorted(card_counter.items())},
            },
            "frequency_buckets": label_dist,
            "frequency_stats": summarize(freq_vals),
            "top_100_cuis": top_100_cuis,
            "bottom_50_cuis": [
                {"cui": c, "freq": f}
                for c, f in cui_counts.most_common()[:-51:-1]
            ],
            "singleton_cui_list": [c for c, f in cui_counts.items() if f == 1],
        },
        "top_cui_overlap_analysis": {
            "description": (
                "Co-occurrence across top-10 CUIs. "
                "High overlap_ratio_of_rarer may indicate synonym-like behavior."
            ),
            "pairs": top_cui_overlap,
        },
        "per_image_entropy": {
            "stats": summarize(per_img_ent),
            "low_entropy_pct": low_entropy_pct,
            "high_entropy_pct": high_entropy_pct,
        },
        "resolution_vs_cui_frequency": resolution_vs_freq,
        "co_occurrence": {
            "min_freq_filter": COOC_MIN_FREQ,
            "total_pairs_after_filter": len(co_list),
            "top_50_by_frequency": co_by_freq[:50],
        },
    }

    print("\n[8/8] Writing output JSONL...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for section, data in stats.items():
            f.write(json.dumps({"section": section, "data": data}, ensure_ascii=False) + "\n")

    print(f"\nDone. Analysis output written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
