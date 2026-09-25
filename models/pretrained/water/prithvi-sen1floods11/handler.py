"""
handler.py — HuggingFace Inference Endpoint handler
Model: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11

Input (JSON POST):
  {
    "bands": {                        # signed COG URLs from perception_fetch_tile
      "B02": "https://...",           # Blue
      "B03": "https://...",           # Green
      "B04": "https://...",           # Red
      "B8A": "https://...",           # Narrow NIR
      "B11": "https://...",           # SWIR 1
      "B12": "https://..."            # SWIR 2
    },
    "bbox": {                         # tile bounding box
      "west": 11.99, "south": 41.4,
      "east": 12.99, "north": 42.4
    },
    "chip_size": 512,                 # optional, default 512
    "confidence_threshold": 0.5       # optional, default 0.5
  }

Output (JSON):
  {
    "dominant_class": "no_flood" | "flood" | "cloud_nodata",
    "flood_pixel_pct": 0.023,         # fraction of valid pixels classified as flood
    "confidence": 0.91,               # mean softmax confidence for dominant class
    "mask_shape": [512, 512],
    "class_counts": {"no_flood": 261144, "flood": 1234, "cloud_nodata": 622},
    "model_id": "Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "model_version": "1.0.0"
  }
"""

import io
import json
import logging
import os
import numpy as np
import torch
import requests
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Band ordering expected by the model (must match training config)
# Sen1Floods11 uses: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2
# ---------------------------------------------------------------------------
BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Class labels
CLASS_LABELS = {
    -1: "cloud_nodata",
     0: "no_flood",
     1: "flood",
}

MODEL_ID      = "Prithvi-EO-2.0-300M-TL-Sen1Floods11"
MODEL_VERSION = "1.0.0"
CHIP_SIZE     = 512


# ---------------------------------------------------------------------------
# Lazy imports — these are available inside the HF endpoint container
# ---------------------------------------------------------------------------
def _import_rasterio():
    import rasterio
    from rasterio.enums import Resampling
    return rasterio, Resampling

def _import_terratorch():
    from terratorch.models import PrithviModelFactory
    from terratorch.registry import FULL_MODEL_REGISTRY
    return PrithviModelFactory, FULL_MODEL_REGISTRY


# ---------------------------------------------------------------------------
# EndpointHandler — HuggingFace Inference Endpoints protocol
# ---------------------------------------------------------------------------

class EndpointHandler:
    def __init__(self, path: str = ""):
        """
        Called once when the endpoint starts.
        `path` is the local directory where model weights are downloaded.
        """
        logger.info(f"🛰️  Loading {MODEL_ID} from {path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"   Device: {self.device}")

        # Load via TerraTorch — it handles the ViT backbone + UNet decoder
        try:
            from terratorch.models import PrithviModelFactory
            factory = PrithviModelFactory()
            self.model = factory.build_model(
                task           = "segmentation",
                backbone       = "prithvi_eo_v2_300",
                decoder        = "UperNetDecoder",
                in_channels    = 6,
                num_classes    = 2,   # flood / no-flood (cloud=-1 masked separately)
                pretrained     = True,
                pretrained_cfg_overlay = {"file": path},
            )
        except Exception as e:
            logger.warning(f"TerraTorch factory failed ({e}), falling back to direct torch.load")
            # Fallback: load weights directly if TerraTorch API changes
            import glob
            ckpt_files = glob.glob(os.path.join(path, "*.pt")) + \
                         glob.glob(os.path.join(path, "*.pth")) + \
                         glob.glob(os.path.join(path, "*.bin"))
            if not ckpt_files:
                raise RuntimeError(f"No checkpoint found in {path}")
            self.model = torch.load(ckpt_files[0], map_location=self.device)

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"✅  {MODEL_ID} loaded on {self.device}")

    # ------------------------------------------------------------------

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Called on every inference request.
        `data` is the parsed JSON body.
        """
        try:
            bands_urls: dict[str, str] = data.get("bands", {})
            bbox:       dict           = data.get("bbox", {})
            chip_size:  int            = int(data.get("chip_size", CHIP_SIZE))
            conf_thr:   float          = float(data.get("confidence_threshold", 0.5))

            if not bands_urls:
                return {"error": "Missing 'bands' field with COG URLs"}

            # 1. Fetch + stack bands into a (6, H, W) float32 array
            chip = self._fetch_chip(bands_urls, chip_size)

            # 2. Run inference
            mask, confidence = self._infer(chip, conf_thr)

            # 3. Summarise
            return self._summarise(mask, confidence)

        except Exception as e:
            logger.exception("Inference error")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_chip(self, bands_urls: dict[str, str], chip_size: int) -> np.ndarray:
        """
        Download each band COG via HTTP range request (window read),
        resample to chip_size × chip_size, stack into (6, H, W) float32.
        """
        rasterio, Resampling = _import_rasterio()
        import rasterio
        from rasterio.enums import Resampling

        chip_bands = []

        for band_name in BAND_ORDER:
            url = bands_urls.get(band_name)
            if not url:
                raise ValueError(f"Missing band URL: {band_name}")

            with rasterio.open(url) as src:
                # Read first overview that covers chip_size resolution
                data = src.read(
                    1,
                    out_shape=(1, chip_size, chip_size),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

                # Sentinel-2 L2A reflectance scaling: divide by 10000
                data = data / 10000.0

                # Clip to [0, 1]
                data = np.clip(data, 0.0, 1.0)

                chip_bands.append(data[0])  # (H, W)

        chip = np.stack(chip_bands, axis=0)  # (6, H, W)
        logger.info(f"   Chip shape: {chip.shape}, mean reflectance: {chip.mean():.4f}")
        return chip

    def _infer(self, chip: np.ndarray, conf_threshold: float):
        """
        Run the model on a (6, H, W) chip.
        Returns (mask: np.ndarray int8, confidence: float).
        """
        # Add batch + time dimensions: model expects (B, T, C, H, W)
        # Sen1Floods11 fine-tune uses T=1 (single timestamp)
        tensor = torch.from_numpy(chip).float()          # (6, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)        # (1, 1, 6, H, W)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)                  # (1, 2, H, W)

        probs = torch.softmax(logits, dim=1)             # (1, 2, H, W)
        pred  = torch.argmax(probs, dim=1)               # (1, H, W)  0=no_flood 1=flood

        # Confidence = mean max probability across all pixels
        max_probs  = probs.max(dim=1).values             # (1, H, W)
        confidence = float(max_probs.mean().cpu())

        mask = pred.squeeze(0).cpu().numpy().astype(np.int8)  # (H, W)
        return mask, confidence

    def _summarise(self, mask: np.ndarray, confidence: float) -> dict:
        """
        Convert (H, W) mask into the GEIANT perception chain result dict.
        """
        h, w = mask.shape
        total_pixels = h * w

        flood_pixels    = int((mask == 1).sum())
        no_flood_pixels = int((mask == 0).sum())
        cloud_pixels    = total_pixels - flood_pixels - no_flood_pixels

        flood_pct = flood_pixels / max(flood_pixels + no_flood_pixels, 1)

        if flood_pct >= 0.10:
            dominant_class = "flood"
        elif cloud_pixels / total_pixels >= 0.50:
            dominant_class = "cloud_nodata"
        else:
            dominant_class = "no_flood"

        return {
            "dominant_class":  dominant_class,
            "flood_pixel_pct": round(flood_pct, 6),
            "confidence":      round(confidence, 6),
            "mask_shape":      [h, w],
            "class_counts": {
                "no_flood":    no_flood_pixels,
                "flood":       flood_pixels,
                "cloud_nodata": cloud_pixels,
            },
            "model_id":      MODEL_ID,
            "model_version": MODEL_VERSION,
        }
