from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shapely.geometry import shape


class Repository:
    def __init__(self, database_path: Path, report_storage=None) -> None:
        self.database_path = database_path
        self.report_storage = report_storage

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON;")
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

                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_updated
                    ON chats(updated_at DESC);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attachments_json TEXT NOT NULL,
                    result_json TEXT,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
                    ON chat_messages(chat_id, created_at);

                CREATE TABLE IF NOT EXISTS chat_images (
                    image_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    url TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_images_chat
                    ON chat_images(chat_id);
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
        if self.report_storage is not None:
            with self._connect() as db:
                geometries_count = db.execute(
                    "SELECT COUNT(*) FROM evidence_geometries WHERE result_id=?", (result_id,)
                ).fetchone()[0]
            run = {
                "result_id": result_id, "task_type": str(task_type), "query_text": query_text,
                "verdict_status": str(verdict_status), "confidence": confidence,
                "created_at": created_at, "geometries_count": geometries_count,
            }
            self.report_storage.save(
                result_id, payload, (output_dir or self.database_path.parent / result_id) / "GeoProof_Report.pdf", run
            )

    def _index_output_geometries(self, result_id: str, output_dir: Path, created_at: str) -> None:
        # Analysis producers may place evidence several directories below the run root.
        geojson_files = sorted(output_dir.rglob("*.geojson"))
        with self._connect() as db:
            db.execute("DELETE FROM evidence_geometries WHERE result_id = ?", (result_id,))
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
        if row:
            return json.loads(row["payload_json"])
        if self.report_storage is not None:
            return self.report_storage.get_result(result_id)
        return None

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

        local_records = [
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
        if self.report_storage is None:
            return local_records
        records = {record["result_id"]: record for record in self.report_storage.list_runs(limit=limit)}
        records.update({record["result_id"]: record for record in local_records})
        merged = list(records.values())
        merged.sort(key=lambda record: record["created_at"], reverse=True)
        if verdict_status:
            merged = [record for record in merged if record["verdict_status"] == verdict_status]
        if task_type:
            merged = [record for record in merged if record["task_type"] == task_type]
        return merged[:limit]

    # =========================================================================
    # Conversation / Chat Persistence Methods
    # =========================================================================

    def create_chat(
        self,
        chat_id: str,
        title: str,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = created_at or datetime.now(UTC).isoformat()
        meta = metadata or {}
        with self._connect() as db:
            db.execute(
                "INSERT INTO chats(chat_id, title, created_at, updated_at, metadata_json) VALUES(?,?,?,?,?)",
                (chat_id, title, now, now, json.dumps(meta)),
            )
        return {
            "chat_id": chat_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "metadata": meta,
        }

    def update_chat(
        self,
        chat_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        updated_at: str | None = None,
    ) -> bool:
        now = updated_at or datetime.now(UTC).isoformat()
        with self._connect() as db:
            row = db.execute("SELECT * FROM chats WHERE chat_id=?", (chat_id,)).fetchone()
            if row is None:
                return False
            curr_title = title if title is not None else row["title"]
            curr_meta = metadata if metadata is not None else json.loads(row["metadata_json"])
            db.execute(
                "UPDATE chats SET title=?, updated_at=?, metadata_json=? WHERE chat_id=?",
                (curr_title, now, json.dumps(curr_meta), chat_id),
            )
        return True

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM chats WHERE chat_id=?", (chat_id,)).fetchone()
            if row is None:
                return None
            return {
                "chat_id": row["chat_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata_json"]),
            }

    def list_chats(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.chat_id, c.title, c.created_at, c.updated_at, c.metadata_json,
                       (SELECT content FROM chat_messages WHERE chat_id=c.chat_id ORDER BY created_at DESC LIMIT 1) as last_message,
                       (SELECT COUNT(*) FROM chat_messages WHERE chat_id=c.chat_id) as message_count,
                       (SELECT COUNT(*) FROM chat_images WHERE chat_id=c.chat_id) as image_count
                FROM chats c
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "chat_id": r["chat_id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "last_message": r["last_message"],
                "message_count": r["message_count"],
                "image_count": r["image_count"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    def delete_chat(self, chat_id: str) -> bool:
        with self._connect() as db:
            res = db.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
            return res.rowcount > 0

    def add_chat_message(
        self,
        message_id: str,
        chat_id: str,
        role: str,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        result: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = created_at or datetime.now(UTC).isoformat()
        att_json = json.dumps(attachments or [])
        res_json = json.dumps(result) if result is not None else None
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO chat_messages(message_id, chat_id, role, content, created_at, attachments_json, result_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                (message_id, chat_id, role, content, now, att_json, res_json),
            )
            db.execute("UPDATE chats SET updated_at=? WHERE chat_id=?", (now, chat_id))
        return {
            "message_id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "created_at": now,
            "attachments": attachments or [],
            "result": result,
        }

    def get_chat_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM chat_messages WHERE chat_id=? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for r in rows:
            attachments = json.loads(r["attachments_json"])
            for att in attachments:
                url = att.get("url", "")
                if url and re.search(r"\.(tif|tiff)$", url, re.IGNORECASE) and not att.get("preview_url"):
                    att["preview_url"] = re.sub(r"\.(tif|tiff)$", "_preview.png", url, flags=re.IGNORECASE)
            messages.append({
                "message_id": r["message_id"],
                "chat_id": r["chat_id"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
                "attachments": attachments,
                "result": json.loads(r["result_json"]) if r["result_json"] else None,
            })
        return messages

    def add_chat_image(
        self,
        image_id: str,
        chat_id: str,
        filename: str,
        storage_path: Path,
        url: str,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = created_at or datetime.now(UTC).isoformat()
        meta = metadata or {}
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO chat_images(image_id, chat_id, filename, storage_path, url, metadata_json, created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (image_id, chat_id, filename, str(storage_path.resolve()), url, json.dumps(meta), now),
            )
        return {
            "image_id": image_id,
            "chat_id": chat_id,
            "filename": filename,
            "storage_path": str(storage_path.resolve()),
            "url": url,
            "metadata": meta,
            "created_at": now,
        }

    def get_chat_images(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM chat_images WHERE chat_id=? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
        return [
            {
                "image_id": r["image_id"],
                "chat_id": r["chat_id"],
                "filename": r["filename"],
                "storage_path": r["storage_path"],
                "url": r["url"],
                "metadata": json.loads(r["metadata_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
