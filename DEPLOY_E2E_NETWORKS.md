# SatQuery AI: E2E Networks Production Deployment Guide

This guide details the complete deployment process for the **SatQuery AI GeoProof Backend** to **[E2E Networks Cloud](https://myaccount.e2enetworks.com/)**.

---

## Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │                   Frontend Client                      │
 │    (Vercel Edge Network or Local Vite React UI)        │
 │              VITE_API_URL=http://<E2E_IP>:8000         │
 └───────────────────────────┬────────────────────────────┘
                             │
                             │ REST API / CORS (* allowed)
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             E2E Networks Linux Compute Node            │
 │             (Ubuntu 22.04 LTS / 24.04 LTS)             │
 │                                                        │
 │   ┌────────────────────────────────────────────────┐   │
 │   │         Docker Container: satquery-backend     │   │
 │   │  - FastAPI ASGI Engine (Port 8000)             │   │
 │   │  - PyTorch + ONNX Geospatial Inference        │   │
 │   │  - GDAL / Rasterio / OpenCV Processing         │   │
 │   └───────────────────────┬────────────────────────┘   │
 │                           │                            │
 │                           ▼                            │
 │   ┌────────────────────────────────────────────────┐   │
 │   │           Docker Persistent Volume             │   │
 │   │            ("satquery_artifacts")              │   │
 │   │    SQLite Database, Raster Artifacts, PDFs     │   │
 │   └────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────┘
```

---

## 1. Create a Linux Compute Node on E2E Networks

1. Log in to your E2E Networks console: **[https://myaccount.e2enetworks.com/](https://myaccount.e2enetworks.com/)**
2. In the left navigation menu, click **Compute** -> **Nodes** (or **Virtual Compute**).
3. Click **Create Node** (or **Launch Instance**).
4. Configure the instance specifications:
   - **Product Category**: **Linux Cloud** or **Smart Dedicated** (or **GPU Cloud** if you plan to use GPU inference).
   - **Operating System**: **Ubuntu 22.04 LTS (64-bit)** or **Ubuntu 24.04 LTS**.
   - **Recommended Plan**:
     - *Standard*: **C3.8GB** (4 vCPU, 8 GB RAM) — recommended for seamless neural network model execution and multi-band raster tiling.
     - *Minimum*: **C3.4GB** (2 vCPU, 4 GB RAM).
   - **Location / Region**: Select closest region (e.g. Noida / Delhi-NCR or Mumbai / Chennai).
   - **Authentication**:
     - Select your **SSH Key** (recommended) or choose password authentication.
5. **Security Group / Firewall Configuration**:
   Ensure the attached Security Group allows the following inbound ports:
   | Port | Protocol | Source | Purpose |
   | :--- | :--- | :--- | :--- |
   | `22` | TCP | `0.0.0.0/0` | SSH Administration |
   | `8000` | TCP | `0.0.0.0/0` | SatQuery FastAPI Backend API |
   | `80` | TCP | `0.0.0.0/0` | HTTP (Optional for Nginx/Certbot) |
   | `443` | TCP | `0.0.0.0/0` | HTTPS (Optional for SSL) |
6. Click **Create / Launch**.
7. Note down the assigned **Public IP** of your node (e.g., `164.52.xxx.xxx`).

---

## 2. Deploying Backend to Your E2E Node

You have two fast deployment methods:

### Method A: 1-Click Remote Deployment from Local Machine (Recommended)

From your local Windows terminal in the project root:

```powershell
powershell .\scripts\deploy_e2e_remote.ps1 -NodeIp "<YOUR_E2E_PUBLIC_IP>"
```

*Example:*
```powershell
powershell .\scripts\deploy_e2e_remote.ps1 -NodeIp "164.52.120.45" -SshUser "root"
```

If you use an SSH private key file:
```powershell
powershell .\scripts\deploy_e2e_remote.ps1 -NodeIp "164.52.120.45" -KeyPath "C:\Users\<user>\.ssh\id_rsa"
```

**What this automated script does:**
1. Tests SSH connectivity to your E2E node.
2. Archives the backend codebase, pre-trained model bundles, and configs into a streamlined deployment bundle.
3. Securely uploads the bundle to `/opt/satquery` on your E2E node.
4. Automatically runs `scripts/deploy_e2e.sh` on the remote server:
   - Installs Docker & Docker Compose if missing.
   - Configures firewall rules.
   - Builds and boots the production container `satquery-backend` via `docker-compose.e2e.yml`.
   - Polls healthcheck until `http://localhost:8000/health` reports status `ok`.
5. Updates your local `frontend\.env` and `frontend\.env.production` with `VITE_API_URL=http://<YOUR_E2E_PUBLIC_IP>:8000`.

---

### Method B: Direct On-Node Deployment via SSH

If you prefer to SSH into your E2E instance and run the deployment directly:

1. **SSH into your E2E Node**:
   ```bash
   ssh root@<YOUR_E2E_PUBLIC_IP>
   ```

2. **Clone the Repository (or copy files)**:
   ```bash
   git clone https://github.com/SM07675/SatQuery.git /opt/satquery
   cd /opt/satquery
   ```

3. **Run the Automated Setup Script**:
   ```bash
   chmod +x scripts/deploy_e2e.sh
   ./scripts/deploy_e2e.sh
   ```

---

## 3. Verify Backend Status

1. **Health Check**:
   Open in your browser:
   ```
   http://<YOUR_E2E_PUBLIC_IP>:8000/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "environment": "production",
     "models": [...]
   }
   ```

2. **Interactive Swagger Documentation**:
   Open in your browser:
   ```
   http://<YOUR_E2E_PUBLIC_IP>:8000/docs
   ```

---

## 4. Connecting the Frontend

Once your backend is live:

1. In [`frontend/.env`](file:///d:/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Prototype_v1.0/frontend/.env) and [`frontend/.env.production`](file:///d:/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Prototype_v1.0/frontend/.env.production), configure:
   ```env
   VITE_API_URL=http://<YOUR_E2E_PUBLIC_IP>:8000
   ```

2. **Deploy to Vercel**:
   ```powershell
   powershell .\scripts\deploy_vercel.ps1
   ```
   *(Or add `VITE_API_URL` under Vercel Settings -> Environment Variables and redeploy).*

3. The status indicator on the SatQuery dashboard will display **`Connected`** in green!

---

## 5. Maintenance & Useful Commands

On your E2E server:

- **Check container status**:
  ```bash
  docker ps
  ```
- **View live backend logs**:
  ```bash
  docker logs -f satquery-backend
  ```
- **Restart backend**:
  ```bash
  docker restart satquery-backend
  ```
- **Stop backend**:
  ```bash
  cd /opt/satquery && docker compose -f docker-compose.e2e.yml down
  ```
