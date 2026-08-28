#!/usr/bin/env python3
"""VRSBench Evaluation Runner for SatQuery GeoProof.

Evaluates EarthDial-4B-RGB and EarthDial-4B-MS checkpoints across standard
VRSBench evaluation samples (Optical VQA, Multispectral VQA, Visual Grounding,
and Land Cover Verification).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.benchmarks import list_available_datasets, run_vrsbench_suite
from app.services.model_registry import registry


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="VRSBench Evaluation Harness for EarthDial")
    parser.add_argument(
        "--variant",
        choices=["auto", "earthdial-4b-rgb", "earthdial-4b-ms"],
        default="auto",
        help="Target model variant to benchmark (default: auto)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export JSON evaluation report",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" SatQuery GeoProof - VRSBench & EarthDial Evaluation Harness")
    print("=" * 80)
    print(f"Target Variant : {args.variant}")
    print(f"Available Datasets:")
    for ds in list_available_datasets():
        print(f"  - {ds.name}: {ds.description} ({ds.total_samples} samples)")
    print("-" * 80)

    print("[*] Running VRSBench frozen-split evaluation suite...")
    response = await run_vrsbench_suite(
        registry=registry,
        model_variant=args.variant,
        max_samples=args.max_samples,
    )

    print("\n--- SAMPLE LEVEL RESULTS ---")
    print(f"{'ID':<12} | {'CATEGORY':<22} | {'MODEL':<17} | {'SIM':<5} | {'IoU':<5} | {'CONF':<5} | {'STATUS'}")
    print("-" * 80)

    for item in response.results:
        iou_str = f"{item.iou:.2f}" if item.iou is not None else "N/A"
        status_str = "[PASS]" if item.passed else "[FAIL]"
        print(
            f"{item.sample_id:<12} | "
            f"{item.category.value:<22} | "
            f"{item.target_model:<17} | "
            f"{item.semantic_similarity:<5.2f} | "
            f"{iou_str:<5} | "
            f"{item.confidence:<5.2f} | "
            f"{status_str}"
        )

    summary = response.summary
    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples Evaluated       : {summary.total_samples}")
    print(f"Passed Samples                : {summary.passed_samples}")
    print(f"Overall Accuracy              : {summary.accuracy_percent:.1f}%")
    print(f"Optical Sub-Accuracy          : {summary.optical_accuracy:.1f}%")
    print(f"Multispectral Sub-Accuracy    : {summary.multispectral_accuracy:.1f}%")
    print(f"Grounding Mean IoU            : {summary.grounding_mean_iou:.3f}")
    print(f"Mean Semantic Similarity      : {summary.mean_semantic_similarity:.3f}")
    print(f"Mean Confidence               : {summary.mean_confidence:.3f}")
    print(f"Mean Inference Latency        : {summary.mean_latency_ms:.1f} ms")
    print(f"Models Evaluated              : {', '.join(summary.model_variants_evaluated)}")
    print("=" * 80)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(response.model_dump(mode="json"), indent=2), encoding="utf-8")
        print(f"[+] Evaluation report exported to: {out_path}")

    return 0 if summary.accuracy_percent >= 80.0 else 1


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
