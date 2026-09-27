# 🛰️ SatQuery AI: Google Cloud (GCP) Deployment Guide

This guide covers deploying the SatQuery AI backend to **Google Cloud Platform (GCP)** with **Scale-to-Zero** ($0.00 cost when idle).

---

## 💡 Why Google Cloud Run is the Best Choice for Evaluators

1. **Scale-to-Zero ($0.00 cost when idle)**:
   - When no evaluator is browsing the website, Google Cloud shuts down all container instances completely.
   - **You are charged 0 credits and 0 dollars while waiting for evaluators.**
2. **Instant Auto-Wake on Request**:
   - The exact second an evaluator opens the website and triggers an analysis, Cloud Run wakes up instantly and handles the request.
3. **Generous Free Tier**:
   - **2,000,000 requests/month free**
   - 360,000 vCPU-seconds and 180,000 GiB-seconds free every single month.
4. **Native HTTPS & Zero SSL Hassles**:
   - Cloud Run gives you a real Google HTTPS domain (e.g. `https://satquery-backend-xxxxx.a.run.app`).
   - Browser "Mixed Content" errors are 100% eliminated on Vercel.

---

## 🚀 3-Minute Deployment via Google Cloud Shell

You **do not** need to install anything on your Windows computer. Google Cloud provides an in-browser terminal (**Cloud Shell**) that already has `gcloud`, `docker`, and `git` configured.

### Step 1: Open Google Cloud Shell
1. Go to **[console.cloud.google.com](https://console.cloud.google.com/)**.
2. Select your Google Cloud project (or create one named `satquery-ai`).
3. Click the **Activate Cloud Shell** icon `[ >_ ]` in the top-right header bar.
4. A terminal window will open at the bottom of your browser.

---

### Step 2: Clone & Deploy
In the Cloud Shell terminal, paste and run these commands:

```bash
# 1. Clone repository
git clone https://github.com/SM07675/SatQuery.git satquery
cd satquery

# 2. Make deployment script executable and run
chmod +x scripts/deploy_gcp.sh
./scripts/deploy_gcp.sh
```

*(Alternatively, run the `gcloud` command directly):*
```bash
gcloud run deploy satquery-backend \
  --source . \
  --platform managed \
  --region asia-south1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars SATQUERY_ENV=production,SATQUERY_DEVICE=cpu
```

> **Note**: When prompted:
> - `Do you want to enable the Cloud Build & Artifact Registry APIs?` ➔ Type `y` and press **Enter**.
> - `Allow unauthenticated invocations?` ➔ Type `y` and press **Enter**.

---

### Step 3: Copy Your HTTPS URL
Once finished (usually 2–3 minutes), Google Cloud will print:
```text
Service URL: https://satquery-backend-xxxxxxxxxx-el.a.run.app
```
Test it in your browser:
- `https://satquery-backend-xxxxxxxxxx-el.a.run.app/health` ➔ `{"status": "healthy"}`
- `https://satquery-backend-xxxxxxxxxx-el.a.run.app/docs` ➔ Interactive Swagger UI

---

### Step 4: Connect Vercel Frontend
Back on your Windows computer in VS Code / Antigravity:

1. Open [`frontend/.env.production`](frontend/.env.production) and set:
   ```env
   VITE_API_URL=https://satquery-backend-xxxxxxxxxx-el.a.run.app
   ```
2. In [`frontend/vercel.json`](frontend/vercel.json), update the backend rewrites to point to your Cloud Run URL:
   ```json
   {
     "rewrites": [
       { "source": "/api/:path*", "destination": "https://satquery-backend-xxxxxxxxxx-el.a.run.app/api/:path*" },
       { "source": "/health", "destination": "https://satquery-backend-xxxxxxxxxx-el.a.run.app/health" },
       { "source": "/artifacts/:path*", "destination": "https://satquery-backend-xxxxxxxxxx-el.a.run.app/artifacts/:path*" },
       { "source": "/(.*)", "destination": "/index.html" }
     ]
   }
   ```
3. Run the 1-click Vercel deployment:
   ```powershell
   powershell .\scripts\deploy_vercel.ps1
   ```

---

### Step 5: Power Off E2E Networks Instance
Now that your backend is running on Google Cloud Run with **0 cost when idle**:
1. Log in to [myaccount.e2enetworks.com](https://myaccount.e2enetworks.com/).
2. Select your node (`C3-16GB-578`).
3. Click **Actions** ➔ **Power Off**.
4. **Result:** Your E2E credit consumption stops completely, and your project remains 100% accessible to evaluators on Google Cloud!
