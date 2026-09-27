#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_MODELS = ROOT / "deployment" / "huggingface-space" / "models"
NODE_IP = "151.185.58.96"
KEY_PATH = r"C:\Users\sarve\key"

SSH_BASE = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", KEY_PATH, f"root@{NODE_IP}"]
SCP_BASE = ["scp", "-P", "22", "-o", "StrictHostKeyChecking=no", "-i", KEY_PATH]


def run_ssh(cmd: str) -> str:
    res = subprocess.run(SSH_BASE + [cmd], capture_output=True, text=True, check=True)
    return res.stdout.strip()


def run_scp(local_path: Path, remote_dest: str):
    subprocess.run(SCP_BASE + [str(local_path), f"root@{NODE_IP}:{remote_dest}"], check=True)


def get_dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def main():
    print("=" * 60)
    print(f"SatQuery E2E Deployment Engine -> {NODE_IP}")
    print("=" * 60)

    # 1. Ensure remote directories exist
    print("[1/5] Ensuring remote /opt/satquery/models directory...")
    run_ssh("mkdir -p /opt/satquery/models /opt/satquery/artifacts")

    # 2. Sync model directories one by one
    print("\n[2/5] Syncing model checkpoints to E2E server...")
    model_dirs = [d for d in SPACE_MODELS.iterdir() if d.is_dir()]
    # Add satquery_buildings_bundle if present locally
    extra_bundle = ROOT / "models" / "satquery_buildings_bundle"
    if extra_bundle.is_dir() and extra_bundle not in model_dirs:
        model_dirs.append(extra_bundle)

    for d in model_dirs:
        name = d.name
        size_mb = get_dir_size_mb(d)
        print(f"\n -> Processing '{name}' ({size_mb} MB)...")

        # Check if already present on remote
        check_output = run_ssh(f"test -d /opt/satquery/models/{name} && find /opt/satquery/models/{name} -type f | wc -l || echo '0'")
        file_count = int(check_output.strip().split("\n")[-1]) if check_output.strip().split("\n")[-1].isdigit() else 0
        local_files = sum(1 for f in d.rglob("*") if f.is_file())

        if file_count >= local_files and local_files > 0:
            print(f"    [SKIP] Already present on server ({file_count} files).")
            continue

        tar_file = Path(os.environ.get("TEMP", "/tmp")) / f"sat_{name}.tar.gz"
        if tar_file.exists():
            tar_file.unlink()

        print(f"    Archiving '{name}'...")
        subprocess.run(["tar", "-czf", str(tar_file), "-C", str(d.parent), name], check=True)

        tar_mb = round(tar_file.stat().st_size / (1024 * 1024), 2)
        print(f"    Uploading {tar_mb} MB to server...")
        t0 = time.time()
        run_scp(tar_file, f"/opt/satquery/models/{name}.tar.gz")
        elapsed = round(time.time() - t0, 1)
        speed = round(tar_mb / max(elapsed, 0.1), 2)
        print(f"    Uploaded in {elapsed}s ({speed} MB/s). Extracting on server...")

        run_ssh(f"cd /opt/satquery/models && tar -xzf {name}.tar.gz && rm -f {name}.tar.gz")
        tar_file.unlink(missing_ok=True)
        print(f"    [DONE] '{name}' successfully installed.")

    # 3. Upload latest backend code and Dockerfile
    print("\n[3/5] Uploading application code, Dockerfile, and docker-compose...")
    code_tar = Path(os.environ.get("TEMP", "/tmp")) / "sat_code.tar.gz"
    if code_tar.exists():
        code_tar.unlink()

    src_dir = ROOT / "deployment" / "huggingface-space"
    subprocess.run([
        "tar",
        "--exclude=*.venv*",
        "--exclude=__pycache__",
        "--exclude=models",
        "-czf",
        str(code_tar),
        "-C",
        str(src_dir),
        "app",
        "satquery_engine",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
    ], check=True)

    run_scp(code_tar, "/opt/satquery/code.tar.gz")
    run_ssh("cd /opt/satquery && tar -xzf code.tar.gz && rm -f code.tar.gz")
    code_tar.unlink(missing_ok=True)
    print("    [DONE] Application code synchronized.")

    # 4. Build & boot Docker container
    print("\n[4/5] Launching production container with Docker Compose...")
    launch_output = run_ssh("cd /opt/satquery && docker compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true && docker compose -f docker-compose.yml up -d --build")
    print(launch_output)

    # 5. Healthcheck verification
    print("\n[5/5] Polling health check on http://" + NODE_IP + ":8000/health...")
    healthy = False
    for i in range(25):
        time.sleep(4)
        try:
            import urllib.request
            req = urllib.request.Request(f"http://{NODE_IP}:8000/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    body = response.read().decode("utf-8")
                    print(f"Health Response: {body}")
                    healthy = True
                    break
        except Exception:
            print(".", end="", flush=True)

    print()
    if healthy:
        print("=" * 60)
        print("  SatQuery AI Backend Successfully Deployed & Healthy!")
        print("=" * 60)
        print(f"API Base URL : http://{NODE_IP}:8000")
        print(f"Swagger Docs : http://{NODE_IP}:8000/docs")
        print(f"Health Check : http://{NODE_IP}:8000/health")

        # Update local frontend .env
        frontend_env = ROOT / "frontend" / ".env"
        frontend_env_prod = ROOT / "frontend" / ".env.production"
        content = f"VITE_API_URL=http://{NODE_IP}:8000\n"
        frontend_env.write_text(content, encoding="utf-8")
        frontend_env_prod.write_text(content, encoding="utf-8")
        print(f"Updated {frontend_env} with VITE_API_URL=http://{NODE_IP}:8000")
    else:
        print("Backend is still initializing or checking. Recent container logs:")
        logs = run_ssh("docker logs --tail 30 satquery-backend")
        print(logs)


if __name__ == "__main__":
    main()
