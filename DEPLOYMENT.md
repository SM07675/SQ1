# SatQuery Deployment Guide: Hugging Face & Vercel

This repository is configured for turnkey decoupled deployment:
- **Backend API**: Hosted on **Hugging Face Spaces** (Docker SDK, FastAPI, Port 7860).
- **Frontend Dashboard**: Hosted on **Vercel** (Vite + React Single Page Application).

---

## 1. Backend Deployment (Hugging Face Spaces)

The backend runs as a high-performance Docker container listening on port `7860`.

### Step 1: Create a Hugging Face Space
1. Log in to [Hugging Face](https://huggingface.co).
2. Go to **Spaces** -> **Create new Space**.
3. Choose a name (e.g., `satquery-api`).
4. Select **Space SDK**: **Docker** (Blank).
5. Choose hardware: **CPU basic** (free) or **CPU upgrade**.
6. Set visibility to **Public** (or Private).

### Step 2: Connect or Push Code
You have two easy options to deploy:

#### Option A: Direct Git Push to Hugging Face
Clone your Space locally and push the repository, or push directly from your local terminal:
```bash
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>
git push hf main --force
```

#### Option B: Automatic Sync via GitHub Actions
If you prefer pushing only to GitHub:
1. Go to your GitHub repository: `Settings` -> `Secrets and variables` -> `Actions`.
2. Add Repository Secrets:
   - `HF_TOKEN`: Your Hugging Face user access token (with `write` permission from HF Settings -> Access Tokens).
   - `HF_SPACE`: `<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>` (e.g., `SM07675/satquery-api`).
3. Every push to the `main` branch on GitHub will automatically deploy to your Hugging Face Space.

### Step 3: Space Configuration & Storage
1. Go to **Settings** in your Hugging Face Space:
   - **Environment Variables**:
     - `SATQUERY_CORS_ORIGINS`: `https://<YOUR_VERCEL_PROJECT>.vercel.app` (or `*` for testing).
     - `SATQUERY_ENV`: `production`
   - **Persistent Storage** (Optional): Attach a persistent volume at `/data` to preserve uploaded rasters, SQLite records, and generated PDF reports across Space container restarts.
2. Your public backend endpoint will be:
   `https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space`

---

## 2. Frontend Deployment (Vercel)

The frontend is an optimized Vite + React Single Page Application with full interactive mapping, dual PDF reports, evidence inspectors, and spatial chat.

### Step 1: Import Project on Vercel
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Select and import the `SM07675/SatQuery` GitHub repository.

### Step 2: Configure Build Settings
- **Root Directory**: Click "Edit" and select `frontend` (or leave root `./` as `vercel.json` supports both).
- **Framework Preset**: `Vite`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### Step 3: Configure Environment Variables
Under **Environment Variables**, add:
- `VITE_API_URL`: Direct HTTPS URL of your Hugging Face Space (without trailing slash):
  ```
  https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space
  ```

### Step 4: Deploy
Click **Deploy**. Vercel will install dependencies, compile TypeScript, build optimized production assets, and serve your app globally on the Vercel Edge Network.

---

## 3. End-to-End Verification

1. **Verify Backend Health**:
   Visit `https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space/health` in your browser.
   Response should be:
   ```json
   {"status":"healthy","version":"1.0.0"}
   ```
2. **Verify Interactive Swagger Docs**:
   Visit `https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space/docs`.
3. **Verify Frontend UI**:
   Open your Vercel deployment URL (e.g., `https://satquery.vercel.app`).
   - The status badge at top should show `Connected`.
   - Upload sample satellite rasters (Sentinel-2, SAR, or aerial imagery).
   - Run analysis and view the interactive GeoProof spatial results, PDF reports, and follow-up query engine.
