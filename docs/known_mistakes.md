# Known Mistakes And Fixes

## Workflow Input Names

Problem: The workflow originally loaded `subject.jpg` and `target.png`, but the repo inputs are `subject_5 year curly.webp` and `superman.png`.

Fix: Keep builder defaults aligned with checked-in assets, or deploy aliases and regenerate the workflow to match those aliases before queueing.

## ReActor Node Registration

Problem: A pod may have `ComfyUI-ReActor` installed on disk while live `/object_info` still does not list `ReActorFaceSwap`.

Fix: Use the exact class name `ReActorFaceSwap`, then do a real pod/container restart from the SimplePod control plane. Do not repeatedly poll or reinstall unless `scripts/simplepod.py preflight` still fails after restart.

## Missing `inswapper_128.onnx`

Problem: ReActor can be installed but fail at runtime if `models/insightface/inswapper_128.onnx` is missing.

Fix: Add the model to `/app/ComfyUI/models/insightface/inswapper_128.onnx`, then rerun preflight.

## Missing Authenticator Token

Problem: `/object_info` can return 401 even over localhost when `ComfyUI-Authenticator` has no `/app/ComfyUI/login/PASSWORD` token file loaded.

Fix: Run `.venv/bin/python scripts/simplepod.py init-auth`, then restart the ComfyUI backend so the authenticator loads the token.

## Overbuilt First Workflow

Problem: The first workflow tried to use diffusion cleanup and Impact Pack detailer nodes before the base swap was proven. On a 12 GB RTX 3060 pod, that adds unnecessary checkpoint/model pressure.

Fix: Keep the source workflow ReActor-only until `faceswap/final_*` is produced and visually accepted. Add diffusion/detailer cleanup as a second milestone.

## Face Restore Defaults

Problem: A weaker `GFPGANv1.4.pth` visibility value produced a softer face than needed.

Fix: Use full restore visibility plus ReActor FaceBoost as the default. Keep `--no-face-boost` available for plain diagnostic swaps.

## ComfyUI Output Caching

Problem: A rerun can report success without writing a fresh output if ComfyUI caches unchanged nodes and output branches.

Fix: For verification runs, change `--filename-prefix` and/or `--intermediate-prefix`, change an input, or clear the server cache before assuming an output node is broken.

## API Prompt Loaded As UI Workflow

Problem: Loading `faceswap_subject_on_character_api.json` in the ComfyUI browser can produce an empty graph and the error that the workflow has no output nodes.

Fix: Load `faceswap_subject_on_character_ui.json` in the browser. Keep `_api` for `/prompt` automation and `_ui` for manual ComfyUI graph execution.

## UI Automation

Problem: ComfyUI browser automation can get stuck on auth or SPA state when SimplePod ports change.

Fix: Prefer SSH/API checks for setup and use human-in-the-loop browser review only when visual workflow actions are faster or safer.

## Stale SimplePod SSH Endpoint

Problem: `scripts/simplepod.py profile` can fail even with valid credentials if the pod was restarted and SimplePod assigned a new external SSH port.

Fix: Refresh `SIMPLEPOD_SSH_HOST` and `SIMPLEPOD_SSH_PORT` in `.env` from the SimplePod UI, then rerun `.venv/bin/python scripts/simplepod.py profile`.
