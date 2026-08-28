"""Mock model servers for SatQuery GeoProof external AI models.

Implements the POST /infer shared HTTP contract on:
- EarthDial (Port 9001): Remote-sensing VQA & captioning
- CROMA (Port 9002): Sentinel-1/Sentinel-2 radar-optical representation
- TinyCD / Open-CD (Port 9003): Bi-temporal change detection
- RemoteCLIP (Port 9004): Text-to-tile semantic retrieval
"""

from __future__ import annotations

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import sys


def make_handler(model_name: str, port: int):
    class ModelInferHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") == "/infer":
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length)
                try:
                    payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                except Exception:
                    payload = {}

                # Build response according to model contract
                if model_name in ("earthdial", "earthdial-4b-rgb", "earthdial-4b-ms"):
                    query = payload.get("query", "")
                    q_lower = query.lower()
                    boxes = []
                    answer = f"EarthDial ({model_name}) analyzed query '{query}'."
                    confidence = 0.89

                    if "central region" in q_lower or "commercial" in q_lower or "dominant land cover" in q_lower:
                        answer = "urban built-up structures and commercial area"
                        boxes = [[0.2, 0.2, 0.8, 0.8]]
                        confidence = 0.92
                    elif "road" in q_lower or "corridor" in q_lower:
                        answer = "Linear road corridor detected spanning east to west across the center."
                        boxes = [[0.40, 0.05, 0.60, 0.95]]
                        confidence = 0.90
                    elif "nir" in q_lower or "canopy" in q_lower or "vegetation" in q_lower or "northern parcel" in q_lower:
                        answer = "Yes, strong NIR reflection and low red reflectance indicate healthy dense vegetation canopy."
                        boxes = [[0.05, 0.05, 0.45, 0.50]]
                        confidence = 0.94
                    elif "water reservoir" in q_lower or "open water" in q_lower or "reservoir" in q_lower:
                        answer = "Water body located with distinct clear boundary."
                        boxes = [[0.12, 0.10, 0.38, 0.40]]
                        confidence = 0.93
                    elif "construction" in q_lower or "earthwork" in q_lower or "bare soil" in q_lower:
                        answer = "Yes, distinct bare soil and active earthwork clearing are observed."
                        boxes = [[0.55, 0.10, 0.90, 0.50]]
                        confidence = 0.88
                    elif "residential" in q_lower or "residential buildings" in q_lower:
                        answer = "Yes, several residential buildings are visible."
                        boxes = [[0.1, 0.1, 0.4, 0.4]]
                        confidence = 0.91
                    elif "forest" in q_lower and ("water" in q_lower or "larger" in q_lower):
                        answer = "Yes, the forest canopy area exceeds the water coverage."
                        boxes = [[0.05, 0.05, 0.6, 0.6]]
                        confidence = 0.90
                    elif "land use change" in q_lower or ("earlier" in q_lower and "later" in q_lower):
                        answer = "Yes, distinct bare soil and active earthwork clearing are observed."
                        boxes = [[0.2, 0.2, 0.7, 0.7]]
                        confidence = 0.89
                    elif "density" in q_lower and ("increase" in q_lower or "development" in q_lower):
                        answer = "Yes, strong NIR reflection and low red reflectance indicate healthy dense vegetation canopy."
                        boxes = [[0.1, 0.1, 0.5, 0.5]]
                        confidence = 0.92
                    elif "water-logging" in q_lower or "moisture" in q_lower or "agricultural" in q_lower:
                        answer = "Low NDWI values indicate normal soil moisture with no catastrophic water-logging."
                        boxes = []
                        confidence = 0.86
                    else:
                        answer = f"EarthDial ({model_name}) verified remote sensing features across tiles."

                    response_data = {
                        "answer": answer,
                        "confidence": confidence,
                        "token_logprobs": [-0.08, -0.06, -0.04],
                        "supports_claim": True,
                        "boxes": boxes,
                        "evidence": {"features_detected": ["vegetation", "canopy", "water_boundary"]},
                        "metrics": {"resolution_m": 10.0, "vqa_score": confidence},
                    }
                elif model_name == "croma":
                    response_data = {
                        "answer": "CROMA fused Sentinel-1 SAR and Sentinel-2 optical channels with high cross-sensor agreement.",
                        "confidence": 0.86,
                        "evidence": {"sar_polarization": "VV/VH", "sensor_agreement_score": 0.84},
                        "metrics": {"radar_optical_iou": 82.5, "sensor_agreement_percent": 82.5},
                    }
                elif model_name == "change":
                    response_data = {
                        "answer": "TinyCD/Open-CD model identified structural surface change between before and after rasters.",
                        "confidence": 0.85,
                        "evidence": {"change_detected": True, "change_type": "surface_modification"},
                        "metrics": {"changed_percent": 4.12, "f1_score": 0.89},
                    }
                elif model_name == "remoteclip":
                    query = payload.get("query", "")
                    tile_paths = payload.get("tile_paths", [])
                    ranked_tiles = tile_paths[:3] if tile_paths else []
                    response_data = {
                        "answer": f"RemoteCLIP semantic retrieval identified top matching tiles for '{query}'.",
                        "confidence": 0.87,
                        "evidence": {"matched_tiles": ranked_tiles},
                        "metrics": {"top_k": len(ranked_tiles), "similarity_score": 0.88},
                    }
                else:
                    response_data = {
                        "answer": f"{model_name} processed request successfully.",
                        "confidence": 0.80,
                        "evidence": {},
                        "metrics": {},
                    }

                response_bytes = json.dumps(response_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "model": model_name}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            sys.stdout.write(f"[{model_name.upper()} :{port}] {format % args}\n")
            sys.stdout.flush()

    return ModelInferHandler


def run_server(model_name: str, port: int):
    server = HTTPServer(("0.0.0.0", port), make_handler(model_name, port))
    print(f"[*] Started {model_name} service on http://localhost:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="SatQuery GeoProof Mock Model Servers")
    parser.add_argument(
        "--model",
        choices=["all", "earthdial", "croma", "change", "remoteclip"],
        default="all",
        help="Which model service to start (default: all)",
    )
    args = parser.parse_args()

    services = {
        "earthdial": 9001,
        "croma": 9002,
        "change": 9003,
        "remoteclip": 9004,
    }

    if args.model == "all":
        threads = []
        for name, port in services.items():
            t = threading.Thread(target=run_server, args=(name, port), daemon=True)
            t.start()
            threads.append(t)
        print("[+] All 4 mock model services are running (Ports: 9001, 9002, 9003, 9004). Press Ctrl+C to stop.")
        try:
            while True:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            print("\nShutting down mock model services.")
    else:
        run_server(args.model, services[args.model])


if __name__ == "__main__":
    main()
