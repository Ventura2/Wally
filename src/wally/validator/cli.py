"""CLI entry point for trajectory shard inspection, validation, and sampling."""

import argparse
import sys
from pathlib import Path

from wally.validator.inspector import (
    compute_action_distribution,
    inspect_dataset_dir,
    inspect_shard,
    validate_shard,
)
from wally.validator.samples import extract_samples

MIN_TRANSITION_COUNT = 100_000


def _is_shard(path: Path) -> bool:
    return path.is_file() and path.suffix == ".tar"


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    passed = True

    if _is_shard(path):
        info = inspect_shard(path)
        print(f"Shard: {path.name}")
        print(f"  Transitions: {info['transition_count']}")
        print(f"  Episodes:    {info['episode_count']}")
        print(f"  Obs shape:   {info['observation_shape']}")
        print(f"  Action keys: {info['action_keys']}")
        print(f"  Timestamps:  {info['timestamp_range'][0]} .. {info['timestamp_range'][1]}")
        total = info["transition_count"]
    else:
        info = inspect_dataset_dir(path)
        print(f"Dataset: {path}")
        print(f"  Shards:      {info['shard_count']}")
        print(f"  Transitions: {info['total_transitions']}")
        print(f"  Obs shape:   {info['observation_shape']}")
        print(f"  Action keys: {info['action_keys']}")
        print(f"  Timestamps:  {info['timestamp_range'][0]} .. {info['timestamp_range'][1]}")
        if "manifest" in info:
            print("  Manifest:    present")
        total = info["total_transitions"]

    if total < MIN_TRANSITION_COUNT:
        print(f"\nTransition count check: FAIL ({total} < {MIN_TRANSITION_COUNT})")
        passed = False
    else:
        print(f"\nTransition count check: PASS ({total} >= {MIN_TRANSITION_COUNT})")

    if args.actions:
        print("\nAction distribution:")
        dist = compute_action_distribution(path)
        for key, stats in dist.items():
            if stats["type"] == "continuous":
                print(f"  {key}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                      f"min={stats['min']:.4f} max={stats['max']:.4f}")
            else:
                print(f"  {key}: {stats['unique_values']} unique values")
                for val, cnt in list(stats["value_counts"].items())[:5]:
                    print(f"    {val}: {cnt}")

    return 0 if passed else 1


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)

    if _is_shard(path):
        shard_paths = [path]
    else:
        shard_paths = sorted(path.glob("*.tar"))

    all_valid = True
    for shard_path in shard_paths:
        result = validate_shard(shard_path)
        if result["valid"]:
            print(f"PASS: {shard_path.name}")
        else:
            all_valid = False
            print(f"FAIL: {shard_path.name}")
            for err in result["errors"]:
                print(f"  - {err}")

    return 0 if all_valid else 1


def cmd_samples(args: argparse.Namespace) -> int:
    path = Path(args.path)
    saved = extract_samples(path, count=args.count, output_dir=args.output_dir)

    if not saved:
        print("No observations found in shard(s).")
        return 1

    print(f"Saved {len(saved)} samples to {args.output_dir}/")
    for p in saved:
        print(f"  {p.name}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectory shard inspector, validator, and sampler."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser(
        "inspect", help="Inspect shard(s) and report statistics"
    )
    p_inspect.add_argument("path", help="Path to a .tar shard or dataset directory")
    p_inspect.add_argument(
        "--actions", action="store_true", help="Show action distribution"
    )
    p_inspect.set_defaults(func=cmd_inspect)

    # validate
    p_validate = subparsers.add_parser(
        "validate", help="Validate shard(s) for schema correctness"
    )
    p_validate.add_argument("path", help="Path to a .tar shard or dataset directory")
    p_validate.set_defaults(func=cmd_validate)

    # samples
    p_samples = subparsers.add_parser(
        "samples", help="Extract random observation samples as PNGs"
    )
    p_samples.add_argument("path", help="Path to a .tar shard or dataset directory")
    p_samples.add_argument(
        "--count", type=int, default=10, help="Number of samples to extract (default 10)"
    )
    p_samples.add_argument(
        "--output-dir", default="samples_output", help="Output directory for PNGs"
    )
    p_samples.set_defaults(func=cmd_samples)

    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
