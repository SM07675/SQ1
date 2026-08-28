from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from shapely.geometry import shape


class Repository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    raster_path TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    result_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    verdict_status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence_geometries (
                    geometry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL,
                    min_x REAL,
                    min_y REAL,
                    max_x REAL,
                    max_y REAL,
                    area_m2 REAL,
                    crs TEXT,
                    geometry_json TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_geom_bbox
                    ON evidence_geometries(min_x, min_y, max_x, max_y);
                CREATE INDEX IF NOT EXISTS idx_evidence_geom_result
                    ON evidence_geometries(result_id);
                """
            )

    def save_asset(self, asset_id: str, raster_path: Path, payload: dict[str, Any], created_at: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO assets(asset_id,raster_path,payload_json,created_at) VALUES(?,?,?,?)",
                (asset_id, str(raster_path.resolve()), json.dumps(payload), created_at),
            )

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["_raster_path"] = row["raster_path"]
        return payload

    def save_result(self, result_id: str, payload: dict[str, Any], created_at: str, output_dir: Path | None = None) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO results(result_id,payload_json,created_at) VALUES(?,?,?)",
                (result_id, json.dumps(payload), created_at),
            )
            # Record in analysis_runs
            task_type = payload.get("task_plan", {}).get("task", "unknown")
            query_text = payload.get("query", "")
            verdict_status = payload.get("verdict", {}).get("status", "unknown")
            confidence = float(payload.get("verdict", {}).get("confidence", 0.0))
            db.execute(
                """
                INSERT OR REPLACE INTO analysis_runs(result_id, task_type, query_text, verdict_status, confidence, created_at, payload_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                (result_id, str(task_type), query_text, str(verdict_status), confidence, created_at, json.dumps(payload)),
            )

        # Decompose and index all GeoJSON geometries generated during analysis
        if output_dir and output_dir.exists():
            self._index_output_geometries(result_id, output_dir, created_at)

    def _index_output_geometries(self, result_id: str, output_dir: Path, created_at: str) -> None:
        geojson_files = list(output_dir.glob("*.geojson")) + list(output_dir.glob("*/*.geojson"))
        for gfile in geojson_files:
            try:
                data = json.loads(gfile.read_text(encoding="utf-8"))
                features = data.get("features", [])
                crs_str = data.get("properties", {}).get("crs")
                producer = data.get("properties", {}).get("producer", gfile.stem)

                with self._connect() as db:
                    for feat in features:
                        geom = feat.get("geometry")
                        if not geom:
                            continue
                        props = feat.get("properties", {})
                        area_m2 = props.get("area_m2")
                        kind = props.get("kind", gfile.stem)

                        # Compute bounding box
                        try:
                            s = shape(geom)
                            min_x, min_y, max_x, max_y = s.bounds
                        except Exception:
                            min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0

                        db.execute(
                            """
                            INSERT INTO evidence_geometries(
                                result_id, producer, evidence_kind, min_x, min_y, max_x, max_y,
                                area_m2, crs, geometry_json, properties_json, created_at
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                result_id,
                                producer,
                                kind,
                                min_x,
                                min_y,
                                max_x,
                                max_y,
                                area_m2,
                                crs_str,
                                json.dumps(geom),
                                json.dumps(props),
                                created_at,
                            ),
                        )
            except Exception:
                pass

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM results WHERE result_id=?", (result_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def query_geometries(
        self,
        result_id: str | None = None,
        evidence_kind: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM evidence_geometries WHERE 1=1"
        params: list[Any] = []

        if result_id:
            query += " AND result_id = ?"
            params.append(result_id)

        if evidence_kind:
            query += " AND evidence_kind = ?"
            params.append(evidence_kind)

        if bbox:
            min_x, min_y, max_x, max_y = bbox
            # Bounding box intersection test: NOT (max_x < min_x OR min_x > max_x ...)
            query += " AND NOT (max_x < ? OR min_x > ? OR max_y < ? OR min_y > ?)"
            params.extend([min_x, max_x, min_y, max_y])

        query += " ORDER BY geometry_id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as db:
            rows = db.execute(query, params).fetchall()

        records = []
        for r in rows:
            records.append({
                "geometry_id": r["geometry_id"],
                "result_id": r["result_id"],
                "producer": r["producer"],
                "evidence_kind": r["evidence_kind"],
                "area_m2": r["area_m2"],
                "crs": r["crs"],
                "bounds": [r["min_x"], r["min_y"], r["max_x"], r["max_y"]],
                "geometry": json.loads(r["geometry_json"]),
                "properties": json.loads(r["properties_json"]),
            })
        return records

    def list_analysis_runs(
        self,
        verdict_status: str | None = None,
        task_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = """
        SELECT r.result_id, r.task_type, r.query_text, r.verdict_status, r.confidence, r.created_at,
               COUNT(g.geometry_id) as geometries_count
        FROM analysis_runs r
        LEFT JOIN evidence_geometries g ON r.result_id = g.result_id
        WHERE 1=1
        """
        params: list[Any] = []
        if verdict_status:
            query += " AND r.verdict_status = ?"
            params.append(verdict_status)
        if task_type:
            query += " AND r.task_type = ?"
            params.append(task_type)

        query += " GROUP BY r.result_id ORDER BY r.created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as db:
            rows = db.execute(query, params).fetchall()

        return [
            {
                "result_id": row["result_id"],
                "task_type": row["task_type"],
                "query_text": row["query_text"],
                "verdict_status": row["verdict_status"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "geometries_count": row["geometries_count"],
            }
            for row in rows
        ]
