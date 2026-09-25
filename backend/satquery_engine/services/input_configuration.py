"""Metadata-based sensor and pair semantics. Appearance never establishes time."""
from __future__ import annotations

from datetime import datetime, UTC
from satquery_engine.schemas import RasterMetadata


def acquisition_time(value):
    date=datetime.fromisoformat(value.replace("Z","+00:00"))
    return date.replace(tzinfo=UTC) if date.tzinfo is None else date.astimezone(UTC)


def classify_configuration(assets: list[RasterMetadata], pair_type: str = "auto") -> tuple[str, str | None]:
    if len(assets) not in (1, 2):
        return "INVALID", "Please upload one or two supported images."
    if len(assets) == 1:
        if assets[0].modality == "optical" and assets[0].bands == 3:
            return "SINGLE_RGB", None
        return {"optical": "SINGLE_OPTICAL", "multispectral": "SINGLE_MULTISPECTRAL", "sar": "SINGLE_SAR"}.get(assets[0].modality, "UNKNOWN"), None
    a, b = assets
    cross = (a.modality == "sar") != (b.modality == "sar") and "unknown" not in (a.modality, b.modality)
    dated = False
    if a.acquisition_date and b.acquisition_date:
        try:
            dated = acquisition_time(a.acquisition_date) != acquisition_time(b.acquisition_date)
        except ValueError:
            pass
    if pair_type == "optical_sar" or cross:
        if not cross:
            return "PAIR_AMBIGUOUS", "I cannot verify which image is radar imagery from its metadata. Please supply imagery with sensor or polarization metadata."
        return "PAIR_MULTISENSOR_BITEMPORAL" if dated else "PAIR_OPTICAL_SAR", None
    if pair_type == "bi_temporal" or dated:
        return "PAIR_BITEMPORAL_SAR" if a.modality == b.modality == "sar" else "PAIR_BITEMPORAL_OPTICAL", None
    return "PAIR_AMBIGUOUS", "Are these images of the same place at different dates, in before-then-after order?"


def policy_blockers(plan, assets, configuration):
    blockers = []
    if plan.requires_pair and len(assets) != 2:
        blockers.append("Please upload two images to answer this comparison question.")
    if plan.requires_temporal_relationship and configuration not in {"PAIR_BITEMPORAL_OPTICAL", "PAIR_BITEMPORAL_SAR"}:
        blockers.append("A verified pair of images from different dates is required for change analysis.")
    if any(n.tool == "fusion" for n in plan.nodes) and configuration not in {"PAIR_OPTICAL_SAR", "PAIR_MULTISENSOR_BITEMPORAL"}:
        blockers.append("This request needs one optical image and one SAR image with identifiable sensor metadata.")
    if any(n.tool == "buildings" for n in plan.nodes) and any(a.modality not in {"optical", "multispectral"} for a in assets):
        blockers.append("The installed building specialist accepts optical RGB imagery; SAR building counting is unavailable.")
    return blockers


def node_policy_error(node,assets):
    """Run before the node invokes any model; failed independent branches can abstain."""
    if node.tool=="fusion":
        from satquery_engine.services.sar import croma_compatibility
        errors=croma_compatibility(assets)
        return " ".join(errors) if errors else None
    if node.tool in {"spectral","spectral_change"}:
        if node.tool == "spectral" and node.parameters["target"] in {"water", "built-up"} and not node.parameters.get("strict"):
            return None  # Specialist performs its own sensor/radiometry preflight.
        from satquery_engine.services.spectral import TARGET_INDEX
        index=TARGET_INDEX[node.parameters["target"]]
        selected=assets if node.tool=="spectral_change" else assets[:1]
        for asset in selected:
            if asset.modality=="sar": return "Optical spectral indices cannot be calculated from SAR backscatter."
            if index not in asset.available_indices and (node.parameters.get("strict") or node.tool=="spectral_change" or asset.bands!=3):
                return f"I cannot calculate {index.upper()} because the required named spectral bands are missing."
    if node.tool=="buildings" and node.parameters["asset"]>=len(assets):
        return "Building comparison requires two images."
    return None
