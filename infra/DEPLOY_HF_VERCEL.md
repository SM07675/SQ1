# SatQuery deployment

The `deployment/huggingface-space` folder is a Docker Space repository. It contains the API, its local model checkpoints, a Dockerfile, and a Space README. Upload that folder to a new Hugging Face Docker Space. The service listens on port 7860. Set the Space variable `SATQUERY_CORS_ORIGINS` to the exact Vercel origin, such as `https://your-project.vercel.app`.

The `deployment/vercel-frontend` folder is the Vite frontend. Import it as a separate Vercel project. Set `VITE_API_URL` to the direct HTTPS URL of the Space, such as `https://your-name-your-space.hf.space`, before building. No trailing slash. Vercel uses `npm run build` and serves `dist`.

Attach a read-write Hugging Face Storage Bucket at `/data` for durable uploads, analysis results, and SQLite chat history. Without that bucket, those files are lost when the Space restarts.

This deployment is suitable for a public demo. The API currently has no user authentication: CORS limits browser access from other origins but does not prevent direct requests. Do not upload private imagery to a public Space. A private Space needs an authenticated API proxy before the Vercel frontend can call it.

If deployment URLs change, update both environment variables and redeploy the Vercel frontend. Check `https://YOUR-SPACE.hf.space/health` and then upload a small image through the Vercel UI.
