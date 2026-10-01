# Deployment Guide

Local Python is **3.11.9**. `runtime.txt`, the Docker base image (`python:3.11-slim`), and the pinned deployment dependencies target Python 3.11.

## Render

1. Push the CineMatch repository to GitHub. The repository currently keeps the app in the `cinematch/` subdirectory; the included `render.yaml` sets `rootDir: cinematch`.
2. In Render choose **New + > Blueprint**, connect the GitHub repository, and select the branch to deploy.
3. Review the `cinematch` web service from `render.yaml`. Confirm runtime is Python, build command is `pip install -r requirements-deploy.txt`, and start command is `gunicorn app.app:app --workers 1 --timeout 120`.
4. Choose the Free instance for a no-cost demo, then click **Apply**. Render supplies `PORT`; the service listens on it through Gunicorn.
5. Wait for the deploy health check, open the generated `onrender.com` URL, and verify `/api/health` returns `{"status":"ok"}`.
6. Note: Render's free web service sleeps after inactivity. The first request after sleep can take longer while the service wakes.

## Hugging Face Spaces

1. In Hugging Face choose **Create new Space**, set the SDK to **Docker**, choose visibility, and create the Space.
2. Push the contents of the `cinematch/` project directory to the Space repository root so `Dockerfile` and `requirements-deploy.txt` are at its root. Include the tracked `artifacts/` and `data/processed/` files; raw data and notebooks are intentionally excluded from the image.
3. The Dockerfile installs the pinned serving dependencies, copies the app and frozen artifacts, and exposes port `7860`. Wait for the Space build to finish.
4. Open the Space URL and check `/api/health`. No debug environment variable is needed.

## Test from your phone

- Open the deployed HTTPS URL on the phone, not `127.0.0.1` from your computer.
- Confirm the CineMatch header and four tabs fit without horizontal scrolling.
- Search for `Toy Story`, select a result, and request similar movies.
- Enter MovieLens user ID `1` and confirm recommendations render.
- In **Rate and discover**, add and rate five movies, then request recommendations; confirm rated titles are excluded.
- Check that loading feedback appears during requests and that an invalid user ID produces a readable error.

## Power BI screenshots and PBIX

Save the requested screenshots as `reports/dashboard/page-1-overview.png`, `reports/dashboard/page-3-model-comparison.png`, and `reports/dashboard/page-5-cold-start.png`. Commit a `.pbix` only if its size is below 25 MB; otherwise keep it local and commit the guide and screenshots.
