# SatQuery AI: Modal & Vercel Production Deployment Guide

This guide details the complete deployment process for the **SatQuery AI GeoProof Platform**:
- **Backend API**: Hosted on **Modal** (`modal.com`) as a scalable serverless ASGI FastAPI service.
- **Frontend Dashboard**: Hosted on **Vercel** (`vercel.com`) as a high-performance Edge Single Page Application.

---

## Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │                   Vercel Edge Network                  │
 │       (React + Vite + Leaflet + Tailwind/Vanilla CSS)  │
 │              https://satquery.vercel.app               │
 └───────────────────────────┬────────────────────────────┘
                             │
                             │ HTTPS / CORS REST API
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │                     Modal Cloud                        │
 │     (Debian Slim, FastAPI ASGI, PyTorch, GDAL, ONNX)   │
 │   https://sarveshm4444--satquery-api-fastapi-app       │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │                  Modal Persistent Volume               │
 │                   ("satquery-artifacts")               │
 │        SQLite database, GeoTIFF masks, PDF reports     │
 └────────────────────────────────────────────────────────┘
```

---

## 1. Backend Deployment (Modal)

The backend runs as a high-performance serverless container defined in [`modal_app.py`](file:///d:/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Prototype_v1.0/modal_app.py).

### Prerequisites & Spend Limit Configuration
Your Modal profile `sarveshm4444` is already configured. Note that if you encounter:
```text
modal.exception.ResourceExhaustedError: Workspace has exceeded its spend limit
```
This is because Modal sets a default hard safety spend limit of **$1.00** on new workspaces, and current usage is at **$0.998**.

1. Visit the Modal Billing Dashboard:  
   👉 **[https://modal.com/settings/sarveshm4444/billing](https://modal.com/settings/sarveshm4444/billing)**
2. Click **Spend Limit** and adjust it (e.g., set to **$5.00** or **$10.00**).
3. If necessary, add a payment method or activate credits.

### Deploy Command
Run the deployment script from the project root:
```powershell
powershell .\scripts\deploy_modal.ps1
```
Or directly using the virtual environment:
```powershell
.\backend\.venv-integrated\Scripts\python.exe -m modal deploy modal_app.py
```

### Production Endpoints
Once deployed, your backend is live at:
- **API Base URL**: `https://sarveshm4444--satquery-api-fastapi-app.modal.run`
- **Swagger Documentation**: `https://sarveshm4444--satquery-api-fastapi-app.modal.run/docs`
- **Health Check**: `https://sarveshm4444--satquery-api-fastapi-app.modal.run/health`

---

## 2. Frontend Deployment (Vercel)

The frontend is an optimized Vite Single Page Application configured in [`frontend/`](file:///d:/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Prototype_v1.0/frontend).

### Pre-configured Environment
The production environment [`frontend/.env.production`](file:///d:/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Complete_Prototype_v1.0/SatQuery_AI_SIH26167_Prototype_v1.0/frontend/.env.production) is set to:
```env
VITE_API_URL=https://sarveshm4444--satquery-api-fastapi-app.modal.run
```

You have two easy deployment methods:

### Method A: Deploy via Vercel CLI (Recommended)
1. In your terminal, run:
   ```powershell
   powershell .\scripts\deploy_vercel.ps1
   ```
2. If prompted for login:
   - Click the displayed device authorization link (or run `npx vercel login`).
   - Confirm your email/GitHub account.
3. The build will automatically compile and output your production Vercel URL (e.g., `https://satquery.vercel.app`).

### Method B: Deploy via Vercel Web Dashboard (Git Integration)
1. Go to **[https://vercel.com/new](https://vercel.com/new)**.
2. Select and import the GitHub repository: `SM07675/SatQuery`.
3. Configure settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Under **Environment Variables**, add:
   - `VITE_API_URL`: `https://sarveshm4444--satquery-api-fastapi-app.modal.run`
5. Click **Deploy**.

---

## 3. End-to-End Verification

1. **Verify Backend Health**:
   ```bash
   curl https://sarveshm4444--satquery-api-fastapi-app.modal.run/health
   ```
   Expected response:
   ```json
   {"status":"healthy","version":"1.0.0"}
   ```

2. **Verify Frontend Connectivity**:
   - Open your deployed Vercel application URL.
   - The top status badge will show `Connected` with green indicator.
   - Run a test spatial query (e.g., upload Sentinel-2 or optical raster).
   - Verify the Key Findings card reports dominant classes with high confidence (e.g., Vegetation 94%), and the vectorized inference pipeline returns in seconds.
