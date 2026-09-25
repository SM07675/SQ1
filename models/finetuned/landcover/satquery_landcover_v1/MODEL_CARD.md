# Unified SatQuery model card

Version: 1.0.0 (registry 2026.09.22)

Intended use: geospatial decision support for RGB VHR land cover and visible building footprints, plus Sentinel-2 surface-water mapping. Outputs are model estimates, not survey-grade inventories.

The exact model, sensor, band, normalization, resolution, threshold, tiling, checkpoint, class, and provenance contracts are machine-readable in `model_registry.json`. Reported numbers come from the existing bundle artifacts and are not recomputed or represented as target-domain accuracy.

Known gaps: RGB land cover used only a bounded FLAIR toy run; RGB building held-out instance F1 is low under a cross-city test; Sentinel-2 land-cover weights are not yet trained; DFC2020 lacks an independent road class; optical water cannot see through thick cloud. Production use requires representative target-domain evaluation by geography and sensor.

