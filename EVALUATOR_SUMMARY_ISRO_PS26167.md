# ⚖️ SIH Evaluator Audit Summary: SatQuery AI vs. ISRO PS-26167

**Document Type:** Executive Technical Audit & Decision Dossier  
**Target System:** SatQuery AI (SIH26167 Prototype v1.0)  
**Evaluator Standard:** Strict Empirical Verification (No credit for pseudocode, mocks, or roadmap promises)  
**Evaluation Date:** September 2026

---

## 📊 1. Overall Audit Scorecard

| Category | Count | Percentage |
| :--- | :---: | :---: |
| 🟢 **1. Implemented & Verified** | **9 / 20** | **45.0%** |
| 🟡 **2. Partially Implemented** | **7 / 20** | **35.0%** |
| 🔵 **3. Proposed / Designed Only** | **3 / 20** | **15.0%** |
| 🔴 **4. Not Implemented / Mock Fallback** | **1 / 20** | **5.0%** |
| **Honest Verified Compliance Score** | — | **62.5%** |

$$
\text{Verified Score} = \frac{9.0 + (7 \times 0.5)}{20} \times 100 = \mathbf{62.5\%}
$$

---

## 🔍 2. Ground-Truth Reality: What is Genuine vs. What is Mocked

### ✅ What is Genuinely Implemented & Verified (Full Credit)
1. **Real Trained Deep Learning Checkpoints (>1.0 GB Total):**
   * `satquery_water_state_dict.pt` (360.3 MB): Swin-v2-B + 6-band spectral fusion head for surface water segmentation.
   * `model/satquery_landcover_v1_bundle/best_checkpoint.pt` (330.3 MB): `mit_b2` U-Net trained on FLAIR-1.
   * `model/satquery_buildings_bundle/satquery_buildings_state_dict.pt` (359.8 MB): Swin-v2-B for building footprint instance detection.
2. **Geospatial Data Engineering:**
   * Full GDAL/Rasterio CRS validation, affine geotransform extraction, and coordinate conflict handling.
   * Fast Fourier Transform (FFT) Phase Correlation and ECC sub-pixel image registration.
3. **Physical Spectral Engines:**
   * Deterministic McFeeters NDWI, Rouse NDVI, and Kawamura NDBI for grounded baseline evidence.
   * Physical Optical–SAR consensus (Optical NDWI confirming low SAR specular radar return).
4. **Outputs & UI:**
   * Vector GeoJSON polygonization with real-world ground area calculations ($m^2$ and hectares).
   * ReportLab PDF audit report generation (`GeoProof_Report.pdf`).
   * Fully functional React 19 + TypeScript + Vite web dashboard (58/58 passing backend pytest tests).

---

### ⚠️ What is Mocked or Simulated (Critical Flaws to Fix)
1. **EarthDial VLM is Not Running Locally:**
   * There are no `earthdial.pt` weights in the repository.
   * In `backend/app/services/model_registry.py:L258-L333`, `_evaluate_vrsbench_fallback()` intercepts queries and returns **hardcoded regex answers** (e.g. if query contains `"commercial"`, it returns `"urban built-up structures and commercial area"` with box `[[0.2, 0.2, 0.8, 0.8]]`).
2. **RemoteCLIP is a Pixel Heuristic:**
   * `compute_tile_semantic_score` uses basic RGB brightness/greenness math ($G - (R+B)/2$) rather than CLIP cross-modal text-image embeddings.
3. **TinyCD & CROMA are Adapter Stubs:**
   * No local weights exist; they fall back to physical spectral/Otsu filters or `scripts/mock_model_servers.py`.
4. **Confidence Calibration Parameters are Static:**
   * Platt scaling sigmoid parameters ($A=5.2, B=-2.4$) are hardcoded rather than fitted dynamically via logistic regression over validation logits.

---

## 📋 3. Complete 20-Point Requirement Matrix

| # | Requirement Dimension | Implementation Reality | Status | Risk Level |
| :-: | :--- | :--- | :---: | :---: |
| 1 | Single-Image RS VQA | Pipeline exists; VLM weights missing (hardcoded regex mock) | **PARTIALLY IMPLEMENTED** | **CRITICAL** |
| 2 | Additional Single-Image Task | Scene description & water/building grounding fully working | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 3 | Bi-Temporal Image Analysis | FFT Phase Correlation + ECC sub-pixel registration | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 4 | Change Detection & Description | SSIM Otsu & spectral deltas work; TinyCD is an adapter stub | **PARTIALLY IMPLEMENTED** | **MEDIUM** |
| 5 | Optical–SAR Complementary Analysis | Physical NDWI ∩ SAR backscatter fusion works; CROMA is stub | **PARTIALLY IMPLEMENTED** | **MEDIUM** |
| 6 | Co-Registered Optical–SAR Processing | Modality detection and spatial grid resampling work | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 7 | Agentic Orchestration | Typed DAG planner compiles intent and selects tools | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 8 | Input Validation | GDAL/Rasterio CRS, bounds, band checks, blockers/warnings | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 9 | Fine-Tuning / Domain Adaptation | 3 real fine-tuned PyTorch vision checkpoints (>1.0 GB) | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 10 | Actual Training Evidence | Checkpoints, configs, metrics exist; raw train script missing | **PARTIALLY IMPLEMENTED** | **MEDIUM** |
| 11 | Actual Model Integration | In-process vision models integrated; VLM/witnesses are stubs | **PARTIALLY IMPLEMENTED** | **HIGH** |
| 12 | Evidence Validation & Arbitration | GeoProof rules enforce missing band & alignment abstention | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 13 | Confidence & Uncertainty | Multi-factor quantitative scoring formula | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 14 | Confidence Calibration on Real Data | Platt sigmoid coded, but coefficients ($A, B$) are static | **PROPOSED / PARTIAL** | **HIGH** |
| 15 | Visual Outputs | Heatmaps, masks, cyan glow, GeoJSON polygons with area | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 16 | Execution Trace & PDF Report | ReportLab engine generates multi-page audit dossiers | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 17 | Working GUI / Web Application | React 19 + TypeScript + Vite (build passed, 58 tests pass) | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 18 | ISRO Cartosat + RISAT Compatibility | GeoTIFF GRD/Optical band parsing works | **IMPLEMENTED AND VERIFIED** | **LOW** |
| 19 | Benchmark Evaluation & Baselines | Water model evaluated; VLM evaluated on circular mocks | **PARTIALLY IMPLEMENTED** | **HIGH** |
| 20 | End-to-End Working Demonstration | 58 passing tests, live demo script, full stack runnable | **IMPLEMENTED AND VERIFIED** | **LOW** |

---

## 🚦 4. Final Evaluator Verdict

> **Can this project currently claim complete compliance with ISRO PS-26167?**  
> ### **NO. It cannot claim complete compliance today.**

### Evaluator Justification:
The project demonstrates an **outstanding geospatial engineering foundation** with genuine raster validation, sub-pixel registration, 3 legitimate fine-tuned PyTorch checkpoints, vector GeoJSON export, ReportLab PDF reporting, and a responsive web GUI.

However, **the headline Vision-Language Model (`EarthDial-4B`) does not run locally; queries are intercepted by `_evaluate_vrsbench_fallback()` returning canned regex mock strings.** An evaluator asking an spontaneous question during the live demo will immediately uncover these fallback stubs.

---

## 🎯 5. Actionable Remediation Plan (To Reach 100%)

1. **Deploy a Real Lightweight Local VLM (Priority 1):**  
   Run an open-source multimodal model (such as `Qwen2-VL-2B-Instruct` or `PaliGemma-3B`) via Ollama, vLLM, or HuggingFace on port 9010. Remove `_evaluate_vrsbench_fallback()` so questions receive dynamic, genuine reasoning.
2. **Be Transparent in Presentation (Priority 2):**  
   Highlight `SatlasWaterNet`, `satquery_landcover`, and `satquery_buildings` as your primary deep learning achievements. Accurately designate change detection and optical–SAR tools as **"deterministic physical multi-sensor consensus engines"** rather than claiming CROMA and TinyCD neural networks are running locally.
3. **Fit Platt Calibration Coefficients (Priority 3):**  
   Run a quick logistic regression over `satquery_water_metrics.json` validation logits to replace the static $A, B$ constants with fitted values.
