from __future__ import annotations

import argparse
import json

from .inference import InferenceOptions, run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SatQuery land-cover, water, and building pipeline.")
    parser.add_argument("input", help="Input georeferenced RGB or 13-band Sentinel-2 GeoTIFF")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--band-order", nargs="+", help="Explicit source band order; required for 13-band files without descriptions")
    parser.add_argument("--cloud-mask", help="Aligned single-band mask where nonzero pixels are cloud/invalid")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--require-all-models", action="store_true", help="Fail instead of emitting an explicit partial result when a sensor-specific model is untrained")
    parser.add_argument("--no-building-override", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(
        args.input, args.output, band_order=args.band_order, cloud_mask_path=args.cloud_mask,
        options=InferenceOptions(device=args.device, mixed_precision=not args.no_amp, require_all_models=args.require_all_models, building_overrides_landcover=not args.no_building_override),
    )
    print(json.dumps({"building_count": result["building_count"], "warnings": result["warnings"], "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
