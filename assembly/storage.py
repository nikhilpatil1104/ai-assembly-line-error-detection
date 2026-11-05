"""SQLite inspection history, single image overlays, batch continuity, CSV exports."""
import csv
import io
import json
import sqlite3
from pathlib import Path


class InspectionStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_file = self.directory / "inspections.sqlite3"
        self.overlays = self.directory / "annotated"
        self.overlays.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _conn(self):
        connection = sqlite3.connect(self.db_file, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._conn() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inspections (
                inspection_id TEXT PRIMARY KEY,
                batch_id TEXT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                detector_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                reasons TEXT NOT NULL,
                detections TEXT NOT NULL,
                findings TEXT NOT NULL,
                processing_ms REAL NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_batch ON inspections(batch_id)")

    def save(self, result: dict, overlay: bytes | None):
        findings = {k: result.get(k) for k in (
            "reason_codes", "missing", "misplaced", "extra", "unknown",
            "uncertain", "detected_counts", "processing_error", "assembly_id",
            "confidence_note"
        )}
        with self._conn() as db:
            db.execute("""INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                result["inspection_id"], result.get("batch_id"), result["created_at"],
                result["filename"], result["detector_mode"], result["status"],
                json.dumps(result["reasons"]), json.dumps(result["detections"]),
                json.dumps(findings), result["processing_ms"],
            ))
        if overlay is not None:
            (self.overlays / (result["inspection_id"] + ".png")).write_bytes(overlay)

    @staticmethod
    def _deserialize(row):
        record = dict(row)
        record["reasons"] = json.loads(record["reasons"])
        record["detections"] = json.loads(record["detections"])
        record.update(json.loads(record.pop("findings")))
        record["annotated_image_url"] = (
            None if record.get("processing_error") else
            f"/inspections/{record['inspection_id']}/annotated"
        )
        return record

    def list(self, limit=100):
        with self._conn() as db:
            rows = db.execute("SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._deserialize(row) for row in rows]

    def by_id(self, inspection_id):
        with self._conn() as db:
            row = db.execute("SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
        return self._deserialize(row) if row else None

    def batch(self, batch_id):
        with self._conn() as db:
            rows = db.execute("SELECT * FROM inspections WHERE batch_id = ? ORDER BY created_at", (batch_id,)).fetchall()
        return [self._deserialize(row) for row in rows]

    def summary(self):
        with self._conn() as db:
            count = db.execute("SELECT count(*) FROM inspections").fetchone()[0]
            flagged = db.execute("SELECT count(*) FROM inspections WHERE status != 'Pass'").fetchone()[0]
        return {"inspected_units": count, "flagged_units": flagged,
                "passed_units": count - flagged, "flagged_rate": round(flagged / count, 4) if count else 0.0}

    def export_csv(self):
        out = io.StringIO()
        fields = ("inspection_id", "batch_id", "created_at", "filename", "detector_mode", "status",
                  "reason_codes", "reasons", "missing", "misplaced", "uncertain",
                  "processing_error", "processing_ms")
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for record in self.list(limit=100_000):
            row = {field: record.get(field, "") for field in fields}
            for field in ("reason_codes", "reasons", "missing", "misplaced"):
                row[field] = json.dumps(row[field])
            writer.writerow(row)
        return out.getvalue()
