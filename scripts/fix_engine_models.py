import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tar_path = Path("sat_engine_models.tar.gz")

subprocess.run(["tar", "-czf", str(tar_path), "-C", str(ROOT / "backend" / "satquery_engine"), "models"], check=True)
subprocess.run(["scp", "-P", "22", "-o", "StrictHostKeyChecking=no", "-i", r"C:\Users\sarve\key", str(tar_path), "root@151.185.58.96:/opt/satquery/sat_engine_models.tar.gz"], check=True)
tar_path.unlink()

ssh_cmd = "cd /opt/satquery/satquery_engine && tar -xzf ../sat_engine_models.tar.gz && rm -f ../sat_engine_models.tar.gz && docker cp /opt/satquery/satquery_engine/models satquery-backend:/app/satquery_engine/ && docker restart satquery-backend"
subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-i", r"C:\Users\sarve\key", "root@151.185.58.96", ssh_cmd], check=True)
print("SUCCESS: satquery_engine/models updated and container restarted!")
