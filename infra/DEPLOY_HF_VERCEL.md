# SatQuery deployment

The `deployment/huggingface-space` folder is a Gradio SDK Space repository that launches the existing FastAPI service from `app.py` on port 7860. It contains the API and local model checkpoints. Create a blank Gradio Space and upload this folder. This custom Python use of the Gradio SDK is an unofficial Hugging Face workflow. Set the Space variable `SATQUERY_CORS_ORIGINS` to the exact Vercel origin, such as `https://your-project.vercel.app`.

The `deployment/vercel-frontend` folder is the Vite frontend. Import it as a separate Vercel project. Set `VITE_API_URL` to the direct HTTPS URL of the Space, such as `https://your-name-your-space.hf.space`, before building. No trailing slash. Vercel uses `npm run build` and serves `dist`.

Attach a read-write Hugging Face Storage Bucket at `/data` for durable uploads, analysis results, and SQLite chat history. Then set `SATQUERY_ARTIFACT_DIR=/data/artifacts` and `SATQUERY_DATABASE_PATH=/data/artifacts/satquery.sqlite3` in Space variables. Without that bucket and those settings, files are lost when the Space restarts.

ZeroGPU selection does not automatically accelerate FastAPI model calls. The API will be CPU-bound unless adapted to ZeroGPU's decorator-based GPU execution. Build success and full model performance on ZeroGPU have not been verified.

This deployment is suitable for a public demo. The API currently has no user authentication: CORS limits browser access from other origins but does not prevent direct requests. Do not upload private imagery to a public Space. A private Space needs an authenticated API proxy before the Vercel frontend can call it.

If deployment URLs change, update both environment variables and redeploy the Vercel frontend. Check `https://YOUR-SPACE.hf.space/health` and then upload a small image through the Vercel UI.
