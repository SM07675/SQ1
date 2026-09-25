"""SAR radiometry contract with improved speckle filtering and adaptive thresholding."""
from dataclasses import dataclass
import numpy as np
import rasterio
from scipy import ndimage as ndi
from satquery_engine.services.bands import detect_band_map, normalize


@dataclass
class SARData:
    db: np.ndarray
    valid: np.ndarray
    configuration: dict


class SARPreprocessor:
    """Enhanced SAR preprocessing with multi-stage speckle filtering and dual-pol support."""

    def process(self, path, polarizations=("vv", "vh")):
        with rasterio.open(path) as src:
            bands = detect_band_map(src)

            # Gracefully downgrade to single-pol if VH is unavailable
            available_pols = tuple(p for p in polarizations if p in bands.indices)
            if not available_pols:
                raise ValueError("SAR preprocessing requires confirmed polarization bands.")

            tags = {normalize(k): normalize(v) for k, v in src.tags().items()}
            units = tags.get("units", tags.get("unit", tags.get("sarunits", "")))
            representation = tags.get("representation", tags.get("sampletype", ""))
            if units in {"db", "decibel", "decibels"}:
                method = "db"
            elif units in {"linear", "intensity", "power", "amplitude"}:
                method = "amplitude" if representation == "amplitude" or units == "amplitude" else "intensity" if representation in {"intensity", "power", "sigma0", "gamma0", "beta0"} or units in {"intensity", "power"} else None
            else:
                method = None
            if method is None:
                raise ValueError("SAR units/representation are unknown. Confirm dB, linear intensity, or linear amplitude before analysis.")
            indexes = [bands.indices[p] for p in available_pols]
            values = src.read(indexes, masked=True).astype("float32").filled(np.nan)
            for i, index in enumerate(indexes):
                values[i] = values[i] * src.scales[index - 1] + src.offsets[index - 1]
            valid = np.all(np.isfinite(values), axis=0)
            if method != "db":
                valid &= np.all(values > 0, axis=0)
                values = (20 if method == "amplitude" else 10) * np.log10(np.where(values > 0, values, np.nan))
            values[:, ~valid] = np.nan
            if not valid.any():
                raise ValueError("SAR image has no valid backscatter pixels.")

            # Enhanced speckle filtering: multi-stage approach
            # Stage 1: 5x5 median filter for impulse noise removal
            filtered = np.empty_like(values)
            for i in range(values.shape[0]):
                filtered[i] = ndi.median_filter(np.where(valid, values[i], 0.0), size=5)
            # Stage 2: 3x3 mean filter for residual speckle smoothing
            for i in range(filtered.shape[0]):
                filtered[i] = ndi.uniform_filter(filtered[i], size=3)

            # Compute dual-pol ratio if both VV and VH available
            dual_pol_ratio = None
            if len(available_pols) >= 2 and "vv" in available_pols and "vh" in available_pols:
                vv_idx = list(available_pols).index("vv")
                vh_idx = list(available_pols).index("vh")
                # VH/VV ratio in dB: water has low ratio, vegetation has high ratio
                dual_pol_ratio = filtered[vh_idx] - filtered[vv_idx]

            return SARData(filtered, valid, {
                "version": "sar_units_v2",
                "input_representation": method,
                "output_units": "dB",
                "polarizations": list(available_pols),
                "band_indices": indexes,
                "valid_fraction": float(valid.mean()),
                "range_db": [float(np.nanmin(filtered)), float(np.nanmax(filtered))],
                "speckle_filter": "median5x5_then_mean3x3",
                "dual_pol_ratio_available": dual_pol_ratio is not None,
                "dual_pol_ratio": dual_pol_ratio,
            })


def croma_compatibility(assets):
    sar = next((a for a in assets if a.modality == "sar"), None)
    optical = next((a for a in assets if a.modality in {"optical", "multispectral"}), None)
    if sar is None or optical is None:
        return ["CROMA requires identifiable optical and SAR assets."]
    errors = []
    if "sentinel1" not in normalize(sar.sensor or "") or "sentinel2" not in normalize(optical.sensor or ""):
        errors.append("CROMA is unavailable for these sensors: its released representation requires Sentinel-1 and Sentinel-2.")
    if sar.bands != 2 or optical.bands != 12:
        errors.append("CROMA is unavailable for this channel representation: confirmed 2-channel SAR and 12-channel optical data are required.")
    if not {"vv", "vh"} <= set(sar.band_map.get("indices", {})):
        errors.append("CROMA needs confirmed VV and VH channels.")
    # Matching channel counts alone cannot verify order/normalization.
    expected = {"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b9", "b11", "b12"}
    names = {normalize(n).replace("b0", "b") for n in optical.band_names}
    if names != expected:
        errors.append("CROMA optical band order cannot be verified against its Sentinel-2 contract.")
    return errors

