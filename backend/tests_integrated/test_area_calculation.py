from satquery_engine.services.raster import _area_square_meters


def test_geographic_area_is_geodesic_not_degree_squared():
    polygon={"type":"Polygon","coordinates":[[[0,0],[0.01,0],[0.01,0.01],[0,0.01],[0,0]]]}
    area=_area_square_meters(polygon,"EPSG:4326")
    assert area is not None and 1_000_000 < area < 1_500_000
