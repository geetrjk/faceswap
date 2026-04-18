---
name: deploy-ui-review
description: Run the faceswap_deploy UI review loop: verify local Python and Node dependencies, validate the shared stable workflow, execute the safe demo job flow, and capture UI screenshots; optionally queue the real stable workflow only when explicitly asked.
---

# Deploy UI Review

Use this skill when you need a repeatable check of the deployment UI in this repo.

## Default Path

Run the safe review script:

```bash
.venv/bin/python scripts/review_deploy_ui.py
```

What it does:

- checks `.venv` Python packages,
- checks `node_modules` UI packages,
- compiles the backend,
- validates `../faceswap/workflows/stable/visual_prompt_hybrid_v1_{api,ui}.json`,
- runs the FastAPI demo-mode submission flow end to end,
- builds the React frontend,
- captures desktop and mobile UI screenshots under `tmp_ui_review/`.

## Options

- Skip frontend rebuild: `.venv/bin/python scripts/review_deploy_ui.py --skip-build`
- Skip screenshots: `.venv/bin/python scripts/review_deploy_ui.py --skip-screenshots`
- Change screenshot port or folder:

```bash
.venv/bin/python scripts/review_deploy_ui.py --port 4273 --screenshot-dir tmp_ui_review_alt
```

## Live Queue

Only use the live queue path when the user explicitly wants a real backend run and the shared env is ready:

```bash
.venv/bin/python scripts/review_deploy_ui.py --live-queue --live-wait 900
```

That queues the shared stable workflow against the existing ComfyUI backend. Do not use it by default.

## Notes

- UI dependencies are not inside `.venv`; React, Vite, and Playwright live in `node_modules`.
- In this Codex desktop sandbox, local screenshot capture may still require prior approval for a temporary local HTTP server and headless browser launch. Once those command prefixes are approved, the script is reusable without rebuilding the workflow manually each time.
- Do not stop or restart shared ComfyUI services as part of this skill.
