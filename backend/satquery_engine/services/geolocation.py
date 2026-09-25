"""Convert model coordinates to source pixels and then to WGS84.

Pixel coordinates denote pixel edges; callers selecting a pixel centre add 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry


@dataclass(frozen=True)
class ModelTile:
    source_x: float
    source_y: float
    source_width: float
    source_height: float
    model_width: float
    model_height: float

    def to_source_pixel(self, x: float, y: float) -> tuple[float, float]:
        if min(self.source_width, self.source_height, self.model_width, self.model_height) <= 0:
            raise ValueError("Tile dimensions must be positive")
        if not 0 <= x <= self.model_width or not 0 <= y <= self.model_height:
            raise ValueError("Model coordinate is outside the tile")
        return (
            self.source_x + x * self.source_width / self.model_width,
            self.source_y + y * self.source_height / self.model_height,
        )

    def geometry_to_source_pixels(self, geometry: BaseGeometry) -> BaseGeometry:
        if geometry.is_empty or not (0 <= geometry.bounds[0] <= geometry.bounds[2] <= self.model_width and
                                     0 <= geometry.bounds[1] <= geometry.bounds[3] <= self.model_height):
            raise ValueError("Model geometry is outside the tile")
        return transform_geometry(
            lambda x, y, z=None: (
                self.source_x + x * self.source_width / self.model_width,
                self.source_y + y * self.source_height / self.model_height,
            ), geometry,
        )


def source_pixel_to_wgs84(
    x: float, y: float, source_transform: Affine, source_crs: CRS | str | None,
) -> tuple[float, float] | None:
    if source_crs is None:
        return None
    native_x, native_y = source_transform @ (x, y)
    lon, lat = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform(native_x, native_y)
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("Transformed coordinate is outside WGS84 bounds")
    return lon, lat


def model_point_to_wgs84(
    x: float, y: float, tile: ModelTile, source_transform: Affine,
    source_crs: CRS | str | None,
) -> tuple[float, float] | None:
    return source_pixel_to_wgs84(*tile.to_source_pixel(x, y), source_transform, source_crs)


def source_geometry_to_wgs84(
    geometry: BaseGeometry, source_transform: Affine, source_crs: CRS | str | None,
) -> BaseGeometry | None:
    if source_crs is None:
        return None
    native = transform_geometry(lambda x, y, z=None: source_transform @ (x, y), geometry)
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    return transform_geometry(transformer.transform, native)


_USER_POINT = re.compile(
    r"\b(?:mark|show|locate|pin)\s+(?:coordinates?\s+)?"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def parse_user_coordinates(query: str) -> tuple[float, float] | None:
    """Explicit text coordinates are latitude, longitude; GeoJSON reverses them."""
    match = _USER_POINT.search(query)
    if match is None:
        return None
    latitude, longitude = map(float, match.groups())
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude must be within [-90, 90] and longitude within [-180, 180]")
    return latitude, longitude
