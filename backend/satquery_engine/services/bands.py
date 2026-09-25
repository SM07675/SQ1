"""Resolve spectral roles from evidence, never from an unqualified band number."""
from dataclasses import dataclass, field, asdict
import re


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


ALIASES = {
    "blue": {"blue"}, "green": {"green"}, "red": {"red"},
    "nir": {"nir", "nearinfrared", "nir08"},
    "narrow_nir": {"narrownir", "nirnarrow", "nir09", "b8a"},
    "swir": {"swir", "swir1", "swir16", "shortwaveinfrared1"},
    "swir2": {"swir2", "swir22", "shortwaveinfrared2"},
    "vv": {"vv"}, "vh": {"vh"}, "hh": {"hh"}, "hv": {"hv"},
}
S2 = {"b2":"blue", "b3":"green", "b4":"red", "b8":"nir", "b8a":"narrow_nir", "b11":"swir", "b12":"swir2"}
LANDSAT_OLI = {"b2":"blue", "b3":"green", "b4":"red", "b5":"nir", "b6":"swir", "b7":"swir2"}
LANDSAT_TM = {"b1":"blue", "b2":"green", "b3":"red", "b4":"nir", "b5":"swir", "b7":"swir2"}


@dataclass
class BandEntry:
    band_name: str
    raster_index: int
    wavelength: float | None = None
    confidence: float = 1.0
    source: str = "metadata"

    def to_dict(self):
        return {**asdict(self), "band_index": self.raster_index}


@dataclass
class BandMap:
    indices: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    wavelengths_nm: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    bands: dict = field(default_factory=dict)

    def has_band(self, name: str) -> bool:
        return name.lower() in self.indices

    def get_index(self, name: str) -> int | None:
        return self.indices.get(name.lower())

    def to_dict(self):
        d = asdict(self)
        d["bands"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.bands.items()}
        return d


def detect_band_map(src):
    result = BandMap()
    tags = {normalize(k): v for k, v in src.tags().items()}
    sensor = normalize(" ".join(str(tags.get(k, "")) for k in ("sensor", "satellite", "platform")))
    declared_order = [item.strip() for item in str(tags.get("bandorder", "")).split(",")]
    if len(declared_order) != src.count:
        declared_order = []
    description_tokens = {normalize(value) for value in (*src.descriptions, *declared_order) if value}
    numbered_descriptions = {f"b{int(match.group(1))}" for token in description_tokens
        if (match := re.fullmatch(r"(?:band|b)0*(\d+)", token))}
    if "sentinel2" not in sensor and (
        "b8a" in description_tokens
        or ({"b10", "b11", "b12"} <= description_tokens and len(description_tokens) >= 8)
        or {"b2", "b3", "b4", "b8", "b11", "b12"} <= numbered_descriptions
    ):
        sensor = f"{sensor} sentinel2_inferred_from_band_names"
    numbered = S2 if "sentinel2" in sensor else LANDSAT_OLI if any(x in sensor for x in ("landsat8", "landsat9")) else LANDSAT_TM if any(x in sensor for x in ("landsat4", "landsat5", "landsat7")) else {}
    candidates = {}
    exact_s2_candidates = {}
    conflicts = set()
    for i in range(1, src.count + 1):
        band_tags = {normalize(k): v for k, v in src.tags(i).items()}
        names = [src.descriptions[i - 1] or ""] + ([declared_order[i - 1]] if declared_order else []) + [band_tags[k] for k in ("name", "commonname", "bandname", "polarization", "polarisation") if k in band_tags]
        # Color interpretation is authoritative for ordinary RGB files.
        color = src.colorinterp[i - 1].name
        if color in ("red", "green", "blue"):
            names.append(color)
        roles = set()
        reasons = []
        for name in names:
            key = normalize(name)
            s2_match = re.fullmatch(r"(?:band|b)0*(1[0-2]|[1-9]|8a)", key)
            if "sentinel2" in sensor and s2_match:
                token = s2_match.group(1)
                exact_name = "b8a" if token == "8a" else f"b{int(token):02d}"
                exact_s2_candidates.setdefault(exact_name, []).append((i, f"Sentinel-2 metadata: {name}"))
            for role, aliases in ALIASES.items():
                if key in aliases:
                    roles.add(role)
                    if role == "swir":
                        roles.add("swir1")
                    reasons.append(f"band {i}: {name}")
            match = re.fullmatch(r"(?:band|b)0*(\d+)", key)
            if match:
                role = numbered.get("b" + match[1])
                if role:
                    roles.add(role)
                    if role == "swir":
                        roles.add("swir1")
                    reasons.append(f"{sensor}: {name}")
        wave = band_tags.get("wavelength", band_tags.get("centralwavelength"))
        unit = normalize(band_tags.get("wavelengthunits", band_tags.get("wavelengthunit", "")))
        if wave is not None and unit in {"nm", "nanometer", "nanometers", "um", "micrometer", "micrometers"}:
            try:
                nm = float(wave) * (1000 if unit in {"um", "micrometer", "micrometers"} else 1)
                result.wavelengths_nm[str(i)] = nm
                for role, lo, hi in [("blue", 450, 510), ("green", 520, 600), ("red", 630, 690), ("narrow_nir", 855, 890), ("nir", 780, 854), ("swir", 1550, 1750), ("swir2", 2080, 2350)]:
                    if lo <= nm <= hi:
                        roles.add(role)
                        if role == "swir":
                            roles.add("swir1")
                        reasons.append(f"wavelength {nm:g} nm")
            except (ValueError, TypeError):
                result.warnings.append(f"Invalid wavelength for band {i}.")
        if len(roles) > 1 and not (roles == {"swir", "swir1"}):
            conflicts.update(roles)
            result.warnings.append(f"Conflicting role evidence for band {i}: {sorted(roles)}.")
        for role in roles:
            candidates.setdefault(role, []).append((i, reasons))
    for role, choices in candidates.items():
        unique_indices = {choice[0] for choice in choices}
        if role in conflicts or len(unique_indices) != 1:
            result.warnings.append(f"Ambiguous {role} mapping; dependent operations are blocked.")
            continue
        idx = next(iter(unique_indices))
        reasons_str = "; ".join(dict.fromkeys(reason for _, reasons in choices for reason in reasons))
        result.indices[role] = idx
        result.confidence[role] = 1.0  # evidence completeness, not a calibrated probability
        result.evidence[role] = choices[0][1]
        wavelength_val = result.wavelengths_nm.get(str(idx))
        result.bands[role] = BandEntry(
            band_name=role.upper(),
            raster_index=idx,
            wavelength=wavelength_val,
            confidence=1.0,
            source=reasons_str or "metadata",
        )
    for role, choices in exact_s2_candidates.items():
        unique_indices = {choice[0] for choice in choices}
        if len(unique_indices) != 1:
            result.warnings.append(f"Ambiguous exact Sentinel-2 {role.upper()} mapping; model inference is blocked.")
            continue
        index = next(iter(unique_indices))
        result.indices[role] = index
        result.confidence[role] = 1.0
        result.evidence[role] = [choice[1] for choice in choices]
        result.bands[role] = BandEntry(
            band_name=role.upper(), raster_index=index, confidence=1.0,
            source="; ".join(choice[1] for choice in choices),
        )
    # Ensure swir and swir1 cross-alias
    if "swir" in result.indices and "swir1" not in result.indices:
        result.indices["swir1"] = result.indices["swir"]
        result.confidence["swir1"] = result.confidence["swir"]
        result.evidence["swir1"] = result.evidence["swir"]
    elif "swir1" in result.indices and "swir" not in result.indices:
        result.indices["swir"] = result.indices["swir1"]
        result.confidence["swir"] = result.confidence["swir1"]
        result.evidence["swir"] = result.evidence["swir1"]
    return result
