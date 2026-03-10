"""CLI: Convert, merge, and compute statistics for datasets.

Usage:
    python scripts/prepare_data.py --dataset anti_uav --raw-dir data/raw/anti_uav
    python scripts/prepare_data.py --merge --datasets anti_uav lrddv2 drone_vs_bird
    python scripts/prepare_data.py --stats --data-dir data/merged
"""

import argparse
from pathlib import Path

from solodet.data.adapters import ADAPTERS
from solodet.data.merge import merge_datasets
from solodet.data.stats import compute_stats, print_stats


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for SoloDet")
    parser.add_argument("--dataset", type=str, choices=list(ADAPTERS.keys()),
                        help="Dataset to convert")
    parser.add_argument("--raw-dir", type=Path, help="Path to raw dataset")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/processed/<dataset>)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to convert")

    parser.add_argument("--merge", action="store_true", help="Merge processed datasets")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to merge (default: all processed)")

    parser.add_argument("--stats", action="store_true", help="Compute dataset statistics")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset dir for stats (default: data/merged)")
    parser.add_argument("--scale-bins", type=Path,
                        default=Path("configs/eval/scale_bins.yaml"),
                        help="Scale bins config")

    args = parser.parse_args()

    processed_dir = Path("data/processed")
    merged_dir = Path("data/merged")

    # Convert a dataset
    if args.dataset:
        if not args.raw_dir:
            parser.error("--raw-dir required when converting a dataset")

        adapter = ADAPTERS[args.dataset]()
        output = args.output_dir or processed_dir / args.dataset

        available = adapter.get_splits(args.raw_dir)
        print(f"Available splits for {args.dataset}: {list(available.keys())}")

        for split in args.splits:
            if split in available:
                print(f"\nConverting {args.dataset}/{split}...")
                adapter.convert(args.raw_dir, output, split)
            else:
                print(f"Skipping {split} (not found)")

    # Merge datasets
    if args.merge:
        print("\nMerging datasets...")
        merge_datasets(processed_dir, merged_dir, args.datasets)

    # Compute stats
    if args.stats:
        data_dir = args.data_dir or merged_dir
        print(f"\nComputing stats for {data_dir}...")
        df = compute_stats(data_dir, args.scale_bins)
        print_stats(df)


if __name__ == "__main__":
    main()
