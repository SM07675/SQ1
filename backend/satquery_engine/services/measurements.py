"""Spectral measurements with explicit proxy names, valid-pixel denominators and GIS outputs."""
import numpy as np
import rasterio
from scipy import ndimage
from satquery_engine.services.spectral import read_index, available_indices, TARGET_INDEX, write_index_png
from satquery_engine.services.spatial_outputs import export_labels


def measure_cover(path, output, target, largest=False, strict=False):
    if target == "water":
        from satquery_engine.services.water_engine import execute_water_pipeline
        return execute_water_pipeline(path, output, largest=largest, strict=strict)

    index = TARGET_INDEX[target]
    exact = index in available_indices(path)
    if target == "built-up" and not exact and not strict:
        from satquery_engine.services.buildings import detect_buildings
        result = detect_buildings(path, output)
        with rasterio.open(output / "buildings_labels.tif") as src:
            valid_pixels = int((src.read_masks(1)>0).sum())
        result.update(method="Building footprint extent (not all built-up surfaces)",
                      coverage_percent=100*result["selected_pixels"]/valid_pixels,
                      valid_pixels=valid_pixels, models_used=[result["model_id"]])
        return result
    with rasterio.open(path) as src:
        if not exact and (strict or src.count != 3):
            raise ValueError(f"I cannot calculate {index.upper()} because the required named spectral bands are missing.")
    values, transform, _ = read_index(path, index, max_size=None, allow_rgb_proxy=not exact)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("No valid pixels are available for this measurement.")

    # Threshold selection: higher for RGB water proxy (prone to false positives)
    if target == "vegetation":
        # Adaptive threshold: use Otsu's method on the finite-value distribution.
        # Clip to [0.12, 0.30] so we stay within defensible spectral territory.
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size > 0:
            hist, bin_edges = np.histogram(finite_vals, bins=200, range=(-1.0, 1.0))
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            w0_c = np.cumsum(hist)
            w1_c = w0_c[-1] - w0_c
            mu0_c = np.cumsum(hist * bin_centres) / np.maximum(w0_c, 1)
            mu1_c = (np.cumsum((hist * bin_centres)[::-1])[::-1]) / np.maximum(w1_c, 1)
            sigma_b = w0_c * w1_c * (mu0_c - mu1_c) ** 2
            otsu_thresh = float(bin_centres[int(np.argmax(sigma_b))])
            threshold = float(np.clip(otsu_thresh, 0.12, 0.30))
        else:
            threshold = 0.20
    elif target == "water" and not exact:
        threshold = 0.20   # RGB proxy needs balanced threshold to capture rivers without land noise
    else:
        threshold = 0.1    # true multispectral NDWI / NDBI

    mask = valid & (values > threshold)

    # --- Vegetation-specific post-processing ---
    quality_warnings: list[str] = []
    if target == "vegetation":
        total_valid = int(valid.sum())
        struct_cross = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_closing(mask, structure=struct_cross, iterations=1)
        mask = ndimage.binary_opening(mask, structure=struct_cross, iterations=1)
        # Sanity gate: if proxy says >60% vegetation, re-threshold stricter
        veg_pct = mask.sum() / max(1, total_valid) * 100
        if veg_pct > 60.0:
            quality_warnings.append(
                f"RGB vegetation proxy initially estimated {veg_pct:.1f}% coverage which is unusually high. "
                "Applying adaptive strictness."
            )
            new_thresh = float(np.clip(threshold + 0.08, 0.20, 0.35))
            mask = valid & (values > new_thresh)
            mask = ndimage.binary_closing(mask, structure=struct_cross, iterations=1)
            mask = ndimage.binary_opening(mask, structure=struct_cross, iterations=1)
            threshold = new_thresh

    # --- Water-specific post-processing (RGB proxy only) ---
    if target == "water" and not exact:
        total_valid = int(valid.sum())

        # A. Morphological closing then opening (cross structure connects bridges & narrow river channels)
        struct_cross = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_closing(mask, structure=struct_cross, iterations=1)
        mask = ndimage.binary_opening(mask, structure=struct_cross, iterations=1)

        # B. Connected component filtering: reject tiny noise clusters while preserving narrow river channels
        min_component_px = max(30, round(total_valid * 0.000025))
        comp_labels, n_comp = ndimage.label(mask)
        if n_comp > 0:
            comp_sizes = np.bincount(comp_labels.ravel())
            comp_sizes[0] = 0  # background
            for cid in range(1, n_comp + 1):
                if comp_sizes[cid] < min_component_px:
                    mask[comp_labels == cid] = False

        # C. Coverage sanity gate: if proxy says > 35% water, re-threshold stricter
        water_pct = mask.sum() / max(1, total_valid) * 100
        if water_pct > 35.0:
            quality_warnings.append(
                f"RGB water proxy initially estimated {water_pct:.1f}% coverage which is unusually high. "
                "Applying adaptive strictness."
            )
            mask = valid & (values > 0.30)
            mask = ndimage.binary_closing(mask, structure=struct_cross, iterations=1)
            mask = ndimage.binary_opening(mask, structure=struct_cross, iterations=1)
            comp_labels, n_comp = ndimage.label(mask)
            if n_comp > 0:
                comp_sizes = np.bincount(comp_labels.ravel())
                comp_sizes[0] = 0
                for cid in range(1, n_comp + 1):
                    if comp_sizes[cid] < min_component_px:
                        mask[comp_labels == cid] = False
            threshold = 0.30  # record the actually-used threshold

    labels, count = ndimage.label(mask)
    if largest and count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        mask = labels == sizes.argmax()
        labels = mask.astype("int32")
    location_desc = None
    if target == "water" and mask.any():
        from satquery_engine.services.spectral import _describe_region_location
        coords = np.argwhere(mask)
        cy = float(coords[:, 0].mean()) / mask.shape[0]
        cx = float(coords[:, 1].mean()) / mask.shape[1]
        location_desc = _describe_region_location(cy, cx)
    name = target.replace("-", "_")
    result = export_labels(labels, path, output, name, transform=transform, valid_mask=valid)
    from satquery_engine.services.spatial_outputs import export_float_raster
    result["paths"].append(export_float_raster(values,path,output/f"{name}_{index if exact else 'rgb_proxy'}.tif"))
    index_path = output / f"{name}_{index if exact else 'rgb_proxy'}.png"
    write_index_png(values, index, index_path)
    result["paths"].append(index_path)
    base_limitations = [] if exact else [f"This is an unvalidated RGB {target} proxy. Shadows and similar colors may be mistaken for {target}."]
    base_limitations.extend(quality_warnings)
    result.update(method=index.upper() if exact else f"RGB {target} proxy", index=index if exact else None,
                  coverage_percent=round(100 * mask.sum() / valid.sum(), 3), valid_pixels=int(valid.sum()),
                  sampled_pixels=int(values.size), threshold=threshold, mean_index=float(values[valid].mean()),
                  evidence_strength=min(0.5 if exact else 0.25, float(np.mean(np.clip(np.abs(values[valid] - threshold) / 0.5, 0, 1)))),
                  threshold_method="fixed_unvalidated_cover_v1", native_resolution=True,
                  limitations=base_limitations, location_description=location_desc)
    return result


def measure_cover_change(before, after, output, target):
    index = TARGET_INDEX[target]
    if index not in available_indices(before) or index not in available_indices(after):
        raise ValueError(f"I cannot measure {target} change reliably because both images need the bands for {index.upper()}.")
    a, transform, _ = read_index(before, index, max_size=None)
    b, _, _ = read_index(after, index, max_size=None)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        raise ValueError("The images have no shared valid pixels.")
    threshold = 0.2 if target == "vegetation" else 0.1
    am, bm = (a > threshold) & valid, (b > threshold) & valid
    gain, loss = bm & ~am, am & ~bm
    labels, _ = ndimage.label(gain | loss)
    result = export_labels(labels, before, output, f"{target.replace('-', '_')}_change", transform=transform)
    for name, mask, color in [("gain", gain, (20,200,100)), ("loss", loss, (240,70,70))]:
        sub = export_labels(ndimage.label(mask)[0], before, output, f"{target.replace('-', '_')}_{name}", transform=transform, color=color)
        result["paths"].extend(sub["paths"])
        result[f"{name}_area_m2"] = sub["area_m2"]
    result.update(method=f"{index.upper()} threshold comparison", before_pixels=int(am.sum()), after_pixels=int(bm.sum()),
                  gain_pixels=int(gain.sum()), loss_pixels=int(loss.sum()), net_pixels=int(bm.sum()-am.sum()),
                  net_percentage_points=float(100*(bm.sum()-am.sum())/valid.sum()),
                  mean_delta_index=float((b-a)[valid].mean()), valid_pixels=int(valid.sum()),
                  evidence_strength=float(np.mean(np.clip(np.abs(b[valid]-a[valid]),0,1))),
                  limitations=["Index changes can reflect season, illumination or atmosphere as well as land-cover changes."])
    return result
