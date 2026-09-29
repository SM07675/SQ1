"""A report stays discoverable and downloadable after the local DB is replaced."""
import json

from app.repository import Repository
from app.services.report_storage import ReportStorage


class Blob:
    def __init__(self, name, objects):
        self.name = name
        self.objects = objects

    def upload_from_filename(self, filename, **_):
        self.objects[self.name] = open(filename, "rb").read()

    def upload_from_string(self, value, **_):
        self.objects[self.name] = value.encode()

    def exists(self):
        return self.name in self.objects

    def download_as_text(self):
        return self.objects[self.name].decode()

    def download_to_filename(self, filename):
        open(filename, "wb").write(self.objects[self.name])


class Bucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return Blob(name, self.objects)

    def list_blobs(self, prefix):
        return [self.blob(name) for name in self.objects if name.startswith(prefix)]


def test_report_survives_database_restart(tmp_path, monkeypatch):
    bucket = Bucket()
    store = ReportStorage("private-report-bucket")
    monkeypatch.setattr(store, "_bucket", lambda: bucket)
    result_id = "d5195903-94ac-4ca8-bbb5-e54360b87e74"
    output = tmp_path / "before" / result_id
    output.mkdir(parents=True)
    (output / "GeoProof_Report.pdf").write_bytes(b"%PDF-1.4\nreport")
    payload = {
        "query": "Find water", "task_plan": {"task": "water"},
        "verdict": {"status": "supported", "confidence": 0.6},
    }
    before = Repository(tmp_path / "before.sqlite", report_storage=store)
    before.init()
    before.save_result(result_id, payload, "2026-09-29T05:00:00+00:00", output)

    after = Repository(tmp_path / "after.sqlite", report_storage=store)
    after.init()
    assert after.get_result(result_id) == payload
    records = after.list_analysis_runs()
    assert len(records) == 1
    assert records[0]["result_id"] == result_id
    assert records[0]["query_text"] == "Find water"
    restored = tmp_path / "after" / "GeoProof_Report.pdf"
    restored.parent.mkdir()
    assert store.restore_pdf(result_id, restored)
    assert restored.read_bytes() == b"%PDF-1.4\nreport"
    assert json.loads(bucket.objects[f"reports/{result_id}/result.json"]) == payload
