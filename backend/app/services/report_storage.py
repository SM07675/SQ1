"""Durable report PDFs and manifests in a private Firebase Storage bucket."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class ReportStorage:
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name

    def _bucket(self):
        from google.cloud import storage

        return storage.Client().bucket(self.bucket_name)

    @staticmethod
    def _id(result_id: str) -> str:
        return str(uuid.UUID(result_id))

    def save(self, result_id: str, payload: dict[str, Any], pdf_path: Path, run: dict[str, Any]) -> None:
        result_id = self._id(result_id)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Report PDF was not generated for {result_id}")
        bucket = self._bucket()
        root = f"reports/{result_id}"
        bucket.blob(f"{root}/GeoProof_Report.pdf").upload_from_filename(
            str(pdf_path), content_type="application/pdf"
        )
        bucket.blob(f"{root}/result.json").upload_from_string(
            json.dumps(payload, separators=(",", ":")), content_type="application/json"
        )
        # Write the index last so the library only shows complete reports.
        bucket.blob(f"report_index/{result_id}.json").upload_from_string(
            json.dumps(run, separators=(",", ":")), content_type="application/json"
        )

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        blob = self._bucket().blob(f"reports/{self._id(result_id)}/result.json")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())

    def restore_pdf(self, result_id: str, destination: Path) -> bool:
        blob = self._bucket().blob(f"reports/{self._id(result_id)}/GeoProof_Report.pdf")
        if not blob.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        return True

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        blobs = self._bucket().list_blobs(prefix="report_index/")
        records = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                records.append(json.loads(blob.download_as_text()))
        records.sort(key=lambda record: record["created_at"], reverse=True)
        return records[:limit]
