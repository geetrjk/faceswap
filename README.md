# faceswap_deploy

Deployment app for the shared SimplePod ComfyUI face-swap stack.

This repo now contains a small FastAPI app that sits in front of the existing `faceswap` workflow assets and queues the stable workflow against the local ComfyUI service on the same SimplePod instance.

## Architecture

- Frontend: polished single-page UI served from this app.
- Backend: FastAPI API that stages inputs, writes metadata to Neon, stores user uploads and outputs in Cloudflare R2, and queues ComfyUI locally.
- Workflow source: `../faceswap/workflows/stable/visual_prompt_hybrid_v1_api.json`
- Shared env source: `../faceswap/.env`

The deployment app intentionally reuses the existing SimplePod setup instead of introducing another workflow host, another env file, or another storage path.

## What Changed

- `app/main.py`: API server and static asset host.
- `app/config.py`: shared-env loader and deployment settings.
- `app/database.py`: Neon-backed job metadata store.
- `app/demo.py`: safe demo-mode fallback for end-to-end UI and API review when shared Neon/R2 keys are not present yet.
- `app/storage.py`: R2 upload helper.
- `app/comfy.py`: local ComfyUI queue and history client.
- `app/workflow.py`: stable workflow patching and template discovery.
- `frontend/*`: React/Vite deployment UI.
- `.agents/skills/deploy-ui-review/SKILL.md`: reusable review skill for this repo.
- `scripts/review_deploy_ui.py`: one-command review runner for validation, demo flow, and screenshots.
- `scripts/run_deploy_app.py`: local app runner.

## Shared Env Contract

The app reads `../faceswap/.env` directly by default.

Existing keys already used by the repo:

```bash
SIMPLEPOD_SSH_HOST=...
SIMPLEPOD_SSH_PORT=...
SIMPLEPOD_SSH_USER=...
SIMPLEPOD_PASSWORD=...
SIMPLEPOD_COMFYUI_URL=...
```

Additional keys expected for this deployment app:

```bash
NEON_DATABASE_URL=postgresql://...
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=...
R2_PUBLIC_BASE_URL=https://cdn.example.com
```

Optional overrides:

```bash
FACE_SWAP_SHARED_ENV=/absolute/path/to/.env
FACE_SWAP_SHARED_ROOT=/absolute/path/to/faceswap
COMFYUI_API_URL=http://127.0.0.1:8188
COMFYUI_INPUT_DIR=/app/ComfyUI/input
COMFYUI_OUTPUT_DIR=/app/ComfyUI/output
COMFYUI_TOKEN_FILE=/app/ComfyUI/login/PASSWORD
STABLE_WORKFLOW_API_PATH=/absolute/path/to/visual_prompt_hybrid_v1_api.json
STABLE_WORKFLOW_UI_PATH=/absolute/path/to/visual_prompt_hybrid_v1_ui.json
TARGET_TEMPLATE_DIR=/absolute/path/to/faceswap
DEPLOY_LOCAL_STATE_DIR=/absolute/path/to/faceswap_deploy/var
```

## Run

Create a virtualenv, install dependencies, then start the app:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
.venv/bin/python scripts/run_deploy_app.py
```

The runner defaults to `0.0.0.0:8000`.

## Review Runner

For the practical repeatable review flow, use:

```bash
.venv/bin/python scripts/review_deploy_ui.py
```

This checks Python and Node dependencies, validates the stable workflow JSONs, runs the demo API flow end to end, builds the React frontend, and captures desktop/mobile UI screenshots under `tmp_ui_review/`.

Optional live queue:

```bash
.venv/bin/python scripts/review_deploy_ui.py --live-queue --live-wait 900
```

Only use `--live-queue` when the user explicitly wants a real ComfyUI run.

## Safe Operating Notes

- This app queues jobs through the local ComfyUI API only.
- It does not stop, restart, or kill ComfyUI processes.
- It stages per-job input filenames to avoid clobbering shared files in the ComfyUI input directory.
- It rewrites `SaveImage` prefixes per job so outputs stay isolated under `faceswap/deploy/<job-id>/...`.
- It uses the checked-in stable workflow from the sibling `faceswap` repo rather than editing shared workflow files in place.

## Verification

Non-disruptive verification for this branch:

```bash
.venv/bin/python scripts/review_deploy_ui.py --skip-screenshots
```
