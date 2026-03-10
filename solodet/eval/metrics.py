"""Per-scale-bin mAP computation via pycocotools."""

import json
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from solodet.utils.config import load_config


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    image_sizes: dict[int, tuple[int, int]],
    scale_bins_config: str | Path | None = None,
) -> dict:
    """Compute mAP overall and per scale bin.

    Args:
        predictions: List of dicts with keys:
            image_id, bbox [x1,y1,w,h], score, category_id
        ground_truths: List of dicts with keys:
            image_id, bbox [x1,y1,w,h], category_id, id
        image_sizes: Mapping of image_id -> (width, height).
        scale_bins_config: Path to scale_bins.yaml.

    Returns:
        Dict with 'overall' and per-bin mAP results.
    """
    if scale_bins_config and Path(scale_bins_config).is_file():
        cfg = load_config(scale_bins_config)
        bins = cfg["bins"]
    else:
        bins = {
            "tiny": {"min": 0, "max": 30},
            "small": {"min": 30, "max": 50},
            "medium": {"min": 50, "max": 100},
            "large": {"min": 100, "max": 100000},
        }

    # Build COCO format
    images = [{"id": img_id, "width": w, "height": h} for img_id, (w, h) in image_sizes.items()]
    categories = [{"id": 0, "name": "drone"}]

    gt_dataset = {
        "images": images,
        "annotations": ground_truths,
        "categories": categories,
    }

    results = {"overall": _run_coco_eval(gt_dataset, predictions)}

    # Per-scale-bin evaluation
    for bin_name, bin_range in bins.items():
        bin_min, bin_max = bin_range["min"], bin_range["max"]

        # Filter GTs by scale
        gt_ids_in_bin = set()
        filtered_gts = []
        for gt in ground_truths:
            _, _, w, h = gt["bbox"]
            max_dim = max(w, h)
            if bin_min <= max_dim < bin_max:
                filtered_gts.append(gt)
                gt_ids_in_bin.add(gt["image_id"])

        if not filtered_gts:
            results[bin_name] = {"mAP50": 0.0, "mAP50_95": 0.0, "n_gt": 0}
            continue

        # Filter predictions to images that have GTs in this bin
        filtered_preds = [p for p in predictions if p["image_id"] in gt_ids_in_bin]

        gt_dataset_bin = {
            "images": [img for img in images if img["id"] in gt_ids_in_bin],
            "annotations": filtered_gts,
            "categories": categories,
        }

        eval_result = _run_coco_eval(gt_dataset_bin, filtered_preds)
        eval_result["n_gt"] = len(filtered_gts)
        results[bin_name] = eval_result

    return results


def _run_coco_eval(gt_dataset: dict, predictions: list[dict]) -> dict:
    """Run COCO evaluation and return mAP metrics."""
    if not predictions or not gt_dataset["annotations"]:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dataset, f)
        gt_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Clean up temp files
    Path(gt_path).unlink(missing_ok=True)
    Path(pred_path).unlink(missing_ok=True)

    return {
        "mAP50_95": float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
        "mAP50": float(coco_eval.stats[1]),      # AP @ IoU=0.50
    }


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n=== Evaluation Results ===\n")
    print(f"{'Scale Bin':>10} | {'mAP@50':>8} | {'mAP@50:95':>10} | {'# GT':>6}")
    print("-" * 45)

    for key in ["overall", "tiny", "small", "medium", "large"]:
        if key not in results:
            continue
        r = results[key]
        n_gt = r.get("n_gt", "—")
        print(f"{key:>10} | {r['mAP50']:>8.3f} | {r['mAP50_95']:>10.3f} | {str(n_gt):>6}")
