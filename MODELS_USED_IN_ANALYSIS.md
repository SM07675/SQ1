# Models & Algorithmic Architectures in SatQuery GeoProof

This document details all machine learning models, neural vision architectures, and deterministic geospatial engines used in **SatQuery GeoProof** for:
1. **Single-Image Water Grounding & Visual Localization**
2. **Before / After Bi-Temporal Change Detection & Multi-Sensor Analysis**

---

## 1. Water Grounding & Spatial Localization Models

```
                                    +-----------------------------------------+
                                    |         Input Image (GeoTIFF/RGB)       |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    | 1. Optical Water Grounding Engine (v2)  |
                                    |    - Multispectral NDWI (Green + NIR)   |
                                    |    - RGB Visible Water Index Proxy      |
                                    |    - Connected Component Segmentation   |
                                    |    - Largest Water Body Isolation       |
                                    +-----------------------------------------+
                                                         |
                                 +-----------------------+-----------------------+
                                 |                                               |
                                 v                                               v
              +-------------------------------------+        +-------------------------------------+
              | 2. RemoteCLIP Tile Retrieval Model   |        | 3. EarthDial Remote-Sensing VLM     |
              |    - Hierarchical Text-to-Tile CLIP |        |    - EarthDial-4B-RGB / 4B-MS       |
              |    - Cosine Semantic Ranking        |        |    - Bounding Box Localization      |
              |    - Spatial NMS (IoU < 0.35)       |        |    - Natural Language Reasoning     |
              +-------------------------------------+        +-------------------------------------+
                                 |                                               |
                                 +-----------------------+-----------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    | 4. GeoProof Multi-Witness Arbiter       |
                                    |    - Calibrated Confidence (Platt)      |
                                    |    - Vector Polygonizer (GeoJSON)       |
                                    |    - Grounding Mask (Electric Cyan)     |
                                    +-----------------------------------------+
```

### A. Optical & Spectral Water Grounding Engine (`optical_water_grounding_engine_v2`)
* **Role:** Deterministic physical water extraction, connected-component analysis, and primary water body delineation.
* **Architecture / Formulation:**
  * **Multispectral Mode:** Calculates McFeeters Normalized Difference Water Index:
    $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
    Threshold: $\text{NDWI} \ge 0.10$ identifies water boundaries.
  * **3-Band Optical RGB Mode:** For standard true-color PNG/JPG/GeoTIFF, uses visible-band absorption physics:
    * Blue/Green reflectance dominance: $\text{Blue} \ge \text{Red} \times 0.85$ and $\text{Green} \ge \text{Red} \times 0.75$.
    * Low surface albedo / dark water constraint: $\text{Brightness} = \frac{R + G + B}{3} < 0.38$.
    * Normalized visible ratio: $\text{NDWI}_{\text{rgb}} = \frac{G - R}{G + R + \epsilon}$, $\text{BlueRatio} = \frac{B - R}{B + R + \epsilon}$.
    * Low-texture surface homogeneity filter: rejects textured urban asphalt and concrete roofs.
  * **Connected Component & Largest Water Body Extraction:**
    * Groups contiguous water pixels into distinct topological components.
    * Computes bounding boxes $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ in normalized $[0, 1]$ and pixel space.
    * Generates `water_grounding_mask.png` with electric cyan fill (`#00e5ff`) and boundary glow (`#38bdf8`), plus vector `water_regions.geojson`.

### B. RemoteCLIP Hierarchical Tile Retrieval Model (`remoteclip_hierarchical_retriever_v1`)
* **Role:** Cross-modal remote-sensing vision-language retrieval.
* **Architecture:** Contrastive Language-Image Pretraining (CLIP) adapted for Earth Observation (RemoteCLIP).
* **Function in Pipeline:**
  * Splits large scenes into overlapping $448 \times 448$ spatial tiles.
  * Encodes the user's query (`"Highlight the largest water body."`) and tile patches into a joint multimodal embedding space.
  * Computes cosine similarity scores between the query vector and candidate tile embeddings.
  * Applies **Spatial Non-Maximum Suppression (NMS)** with $\text{IoU} < 0.35$ to eliminate redundant overlaps and highlight top-$k$ focus areas.

### C. EarthDial Vision-Language Foundation Model (`earthdial-4b-rgb` / `earthdial-4b-ms`)
* **Role:** Remote-sensing VQA, visual grounding, and semantic explanation.
* **Architecture:** 4-Billion parameter multimodal transformer fine-tuned on remote sensing datasets (RSVQA, VRSBench, EarthDial).
* **Function in Pipeline:**
  * Ingests the optical scene and RemoteCLIP-prioritized candidate tiles.
  * Outputs grounded bounding box coordinates `[[ymin, xmin, ymax, xmax]]` and natural language explanatory evidence.

---

## 2. Before / After Bi-Temporal Change Detection Models

```
              +-----------------------+              +-----------------------+
              |    Image A (Before)   |              |    Image B (After)    |
              +-----------------------+              +-----------------------+
                                  \                      /
                                   v                    v
                        +-----------------------------------------+
                        | 1. Sub-Pixel Registration Engine        |
                        |    - Phase Correlation (FFT)            |
                        |    - Enhanced Correlation Coeff (ECC)   |
                        +-----------------------------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
  +-------------------------------------+        +-------------------------------------+
  | 2. Structural & Radiometric Change   |        | 3. Semantic Spectral Change Engine  |
  |    - Structural Similarity (SSIM)   |        |    - Built-up: Delta NDBI (SWIR-NIR)|
  |    - Siamese Difference Features    |        |    - Vegetation: Delta NDVI (NIR-R) |
  |    - Automated Otsu Binarization    |        |    - Water: Delta NDWI (Green-NIR)  |
  +-------------------------------------+        +-------------------------------------+
                     |                                               |
                     v                                               v
  +-------------------------------------+        +-------------------------------------+
  | 4. Learned Change Model (Witness)   |        | 5. CROMA Optical-SAR Fusion Engine  |
  |    - TinyCD / Open-CD Adapter       |        |    - Sentinel-1 SAR + S-2 Optical   |
  |    - Bitemporal Attention Network   |        |    - Cross-Attention Feature Match  |
  +-------------------------------------+        +-------------------------------------+
                     |                                               |
                     +-----------------------+-----------------------+
                                             |
                                             v
                        +-----------------------------------------+
                        | 6. GeoProof Verification & Calibration  |
                        |    - Multi-Witness Consensus Validation |
                        |    - Platt Scaling Confidence [0-100%]  |
                        |    - PDF Audit Report Generation        |
                        +-----------------------------------------+
```

### A. Sub-Pixel Registration & Normalization (`phase_correlation_and_ecc_registration_v1`)
* **Role:** Spatial alignment and radiometric normalization before change deduction.
* **Methods:**
  * Fast Fourier Transform (FFT) **Phase Correlation** for rapid rigid translation discovery.
  * **Enhanced Correlation Coefficient (ECC)** maximization for affine refinement.
  * Measures spatial overlap and alignment score $[0.0, 1.0]$. Blocks analysis if alignment $< 0.15$ to prevent false-positive change artifacts.

### B. Baseline Radiometric & SSIM Change Detector (`baseline_ssim_radiometric_change_detector_v1`)
* **Role:** Domain-agnostic structural surface change detection.
* **Techniques:**
  * **Local Structural Similarity (SSIM):** Compares luminance, contrast, and structural texture across a $7 \times 7$ sliding window.
  * **Multi-Scale Spatial Difference:** Computes directional gradient deltas ($\nabla_x, \nabla_y$).
  * **Automated Otsu Binarization:** Calculates inter-class variance to determine the optimal change decision threshold dynamically.
  * Produces `change_mask.png`, `difference_map.png`, and `change_regions.geojson`.

### C. Learned Deep Change Detection Adapter (`tinycd_and_opencd_adapter_v1`)
* **Role:** Deep feature change representation and cross-scene reasoning.
* **Supported Architectures:**
  * **TinyCD:** Lightweight Siamese convolutional neural network with spatial-temporal attention.
  * **Open-CD / BIT (Bitemporal Image Transformer):** Transformer-based token difference modeling.
  * Acts as an independent witness to confirm or refute radiometric change claims.

### D. Semantic Spectral Change Engine (`deterministic_spectral_proxy_v2`)
* **Role:** Class-specific directional change proof (e.g. *Has built-up increased?*, *Where did vegetation decrease?*).
* **Formulations:**
  * **Built-up / Urban Expansion:**
    $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}, \quad \Delta \text{NDBI} = \text{NDBI}_{T2} - \text{NDBI}_{T1}$$
  * **Vegetation Loss / Deforestation:**
    $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}, \quad \Delta \text{NDVI} = \text{NDVI}_{T2} - \text{NDVI}_{T1}$$
  * **Water Expansion / Flood Mapping:**
    $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}, \quad \Delta \text{NDWI} = \text{NDWI}_{T2} - \text{NDWI}_{T1}$$
  * Produces directional masks (`semantic_change_mask.png`), spectral index heatmaps, and square-metre area measurements.

### E. CROMA Optical-SAR Cross-Attention Fusion (`croma_cross_attention_fusion`)
* **Role:** Joint Sentinel-1 radar and Sentinel-2 optical multi-sensor representation learning.
* **Architecture:** Pretrained CROMA (Cross-Modal Representation Learning for Earth Observation) vision transformer.
* **Mechanism:**
  * Extracts deep embeddings from optical reflectance and radar backscatter ($\text{VV} / \text{VH}$).
  * Computes cosine similarity and IoU consensus mask between radar low-backscatter water bodies and optical NDWI water bodies.

### F. GeoProof Verification & Calibrated Ensemble (`geoproof_spatial_ensemble_v1`)
* **Role:** Final evidence arbiter and uncertainty quantification.
* **Formulation:**
  * Combines input quality score, spatial alignment score, evidence strength, and cross-model agreement.
  * Applies **Platt Scaling** to convert raw ensemble scores into calibrated probabilities with 95% Confidence Intervals $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$ and Expected Calibration Error (ECE $\le 4.2\%$).
  * Enforces safe abstention (`INSUFFICIENT_EVIDENCE` or `DISPUTED`) when evidence is contradictory or ungrounded.

---

## Summary Reference Table

| Task | Primary Models / Engines | Output Artifacts | Output Metrics |
| :--- | :--- | :--- | :--- |
| **Water Grounding** | `optical_water_grounding_engine_v2`, `RemoteCLIP`, `EarthDial-4B-RGB` | `water_grounding_mask.png`, `water_mask.png`, `ndwi.png`, `water_regions.geojson` | Largest water coverage %, Area ($m^2$/ha), Bounding box $[y_1, x_1, y_2, x_2]$ |
| **Bi-Temporal Change** | `SSIM Change Detector`, `TinyCD/Open-CD`, `Deterministic Spectral Delta` | `change_mask.png`, `semantic_change_mask.png`, `difference_map.png`, `change_regions.geojson` | Changed pixel %, Total modified area ($m^2$/ha), Region count |
| **Optical-SAR Fusion** | `CROMA Vision Transformer`, `Optical NDWI + SAR Backscatter Fusion` | `sensor_agreement.png`, `optical_preview.png`, `sar_db_preview.png`, `confirmed_water.geojson` | Sensor agreement IoU %, Cosine feature similarity, Confirmed water area |
| **GeoProof Verdict** | `Platt Calibration Arbiter`, `Multi-Witness Ensemble Verifier` | `audit_report.pdf`, `analysis_manifest.json` | Calibrated probability %, 95% CI, ECE % |
