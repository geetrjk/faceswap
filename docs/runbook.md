# Runbook: ComfyUI Face Swap Workflow

## 1) Local run

### Prerequisites
- Python 3.10+
- NVIDIA GPU + CUDA drivers
- ComfyUI

### Install ComfyUI and required custom nodes

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd custom_nodes
git clone https://github.com/Gourieff/ComfyUI-ReActor.git
cd ../
pip install -r custom_nodes/ComfyUI-ReActor/requirements.txt
python custom_nodes/ComfyUI-ReActor/install.py
```

### Required model files (minimum)

Place in ComfyUI model directories:
- swap model: `models/insightface/inswapper_128.onnx`
- face restore: `models/facerestore_models/GFPGANv1.4.pth`

### Use this repo workflow

From this repo root:

```bash
python scripts/build_faceswap_workflow.py
cp workflows/faceswap_subject_on_character_ui.json /path/to/ComfyUI/user/default/workflows/
```

Then open ComfyUI, load `faceswap_subject_on_character_ui`, set:
- `subject_5 year curly.webp` (human identity source)
- `superman.png` (character target)

Outputs:
- `faceswap/final_*` (base ReActor swap)

## 2) SimplePod run

The repo expects `.env` at the project root. Do not commit it.

Required keys:

```bash
SIMPLEPOD_SSH_HOST=<ip-or-host>
SIMPLEPOD_SSH_PORT=22
SIMPLEPOD_SSH_USER=root
SIMPLEPOD_PASSWORD=<password>
SIMPLEPOD_COMFYUI_URL=<browser-url>
```

Install the local helper dependency:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For a fresh SimplePod setup that installs both the ReActor baseline and InstantID experiments, prefer:

```bash
scripts/setup_simplepod_instantid.sh
```

The script profiles the pod, regenerates workflows, installs remote custom nodes/models, deploys the ReActor, InstantID, and crop-stitch workflows, pauses for the required ComfyUI/SimplePod restart, then runs preflight checks.

The setup helper is designed to be idempotent:
- custom node `git clone` attempts time out and fall back to GitHub archive downloads
- large model downloads use resumable `curl -C -`
- model files are checked by minimum expected size, so partial `.safetensors` or `.onnx` files are not treated as valid
- long install commands stream remote output so stalled downloads are visible quickly

Profile the pod before changing anything:

```bash
.venv/bin/python scripts/simplepod.py profile
```

Regenerate and deploy the workflow/assets:

```bash
.venv/bin/python scripts/build_faceswap_workflow.py
.venv/bin/python scripts/simplepod.py deploy
.venv/bin/python scripts/simplepod.py init-auth
```

For manual browser runs, load `faceswap_subject_on_character_ui` from the `faceswap` workflow folder. Do not load `faceswap_subject_on_character_api` in the browser; that file is the API prompt used by `scripts/simplepod.py queue`.

Verify live ComfyUI dependencies before queueing:

```bash
.venv/bin/python scripts/simplepod.py preflight
```

If `preflight` reports missing custom nodes after they were installed, restart the pod/container from the SimplePod control plane and run `preflight` again. Avoid repeated polling.

Queue the base workflow:

```bash
.venv/bin/python scripts/simplepod.py queue --wait 300
.venv/bin/python scripts/simplepod.py download faceswap/final_00001_.png --local-dir test_outputs
```

The current default workflow uses ReActor FaceBoost with `GFPGANv1.4.pth`. For a plain swap-only diagnostic, regenerate with:

```bash
.venv/bin/python scripts/build_faceswap_workflow.py --no-face-boost --face-restore-visibility 0.7
```

## 2b) Experimental InstantID run

Use this path only for the separate subject-first SDXL + InstantID experiment. It does not replace the ReActor baseline workflow.

```bash
.venv/bin/python scripts/build_instantid_workflow.py
.venv/bin/python scripts/simplepod.py deploy-instantid
.venv/bin/python scripts/simplepod.py preflight-instantid
.venv/bin/python scripts/simplepod.py queue --workflow workflows/instantid_subject_pose_style_api.json --wait 600
```

For the crop-first/stitch-back inspection variant:

```bash
.venv/bin/python scripts/build_instantid_crop_stitch_workflow.py
.venv/bin/python scripts/simplepod.py deploy-instantid-crop
.venv/bin/python scripts/simplepod.py preflight-instantid --crop-stitch
```

For manual browser inspection, load `instantid_crop_stitch_experiment_ui` from the `faceswap` workflow folder. It is experimental and keeps separate checkpoints for the crop region mask, shrunken edit mask, raw crop inpaint, crop composite, and final stitched image.

For the separate ReActor-first swap-and-bake sidecar:

```bash
.venv/bin/python scripts/build_swap_and_bake_workflow.py
.venv/bin/python scripts/simplepod.py deploy-swap-and-bake
.venv/bin/python scripts/simplepod.py queue --workflow workflows/swap_and_bake_experiment_api.json --wait 600
```

For manual browser inspection, load `swap_and_bake_experiment_ui` from the `faceswap` workflow folder. This workflow keeps the current InstantID pipeline unchanged: it first creates a ReActor swap, saves the face mask for inspection, then runs a low-denoise full-image SDXL bake from the swapped image.

For the visual-prompt hybrid sidecar adapted to the current backend:

```bash
.venv/bin/python scripts/build_visual_prompt_hybrid_workflow.py
.venv/bin/python scripts/simplepod.py deploy-visual-prompt-hybrid
.venv/bin/python scripts/simplepod.py queue --workflow workflows/visual_prompt_hybrid_experiment_api.json --wait 900
```

For manual browser inspection, load `visual_prompt_hybrid_experiment_ui` from the `faceswap` workflow folder. The reference design expects PuLID, IP-Adapter Plus, and SAM/Impact nodes; the current live backend does not expose those nodes, so the checked-in sidecar uses the available fallback: FaceSegmentation target head mask, SDXL head fill, ReActor likeness snap, and low-denoise full-image bake.

The current visual-prompt branch keeps exposed-skin harmonization outside the workflow tail. The workflow saves `pre_skin_harmonize` plus `target_skin_mask`, then `scripts/remote_skin_tone_postprocess.py` refines the semantic skin mask, excludes the already-solved face/neck region, and transfers the solved face tone onto exposed non-face skin deterministically.

The InstantID path needs additional custom nodes and models. See `docs/instantid_experiment.md` before provisioning or queueing.

Human-in-the-loop output checkpoints:
- `faceswap/intermediate/subject_input_*`
- `faceswap/intermediate/target_input_*`
- `faceswap/intermediate/plain_swap_*`
- `faceswap/final_*`

InstantID experiment checkpoints:
- `faceswap/instantid/intermediate/subject_identity_*`
- `faceswap/instantid/intermediate/target_pose_style_*`
- `faceswap/instantid/intermediate/target_face_mask_*`
- `faceswap/instantid/intermediate/target_face_keypoints_*`
- `faceswap/instantid/intermediate/face_inpaint_raw_*`
- `faceswap/instantid/final_*`

InstantID crop-stitch experiment checkpoints:
- `faceswap/instantid_crop_stitch/intermediate/target_crop_*`
- `faceswap/instantid_crop_stitch/intermediate/target_crop_region_mask_*`
- `faceswap/instantid_crop_stitch/intermediate/target_crop_edit_mask_*`
- `faceswap/instantid_crop_stitch/intermediate/target_crop_canny_*`
- `faceswap/instantid_crop_stitch/intermediate/crop_inpaint_raw_*`
- `faceswap/instantid_crop_stitch/intermediate/crop_composite_*`
- `faceswap/instantid_crop_stitch/final_*`

Swap-and-bake experiment checkpoints:
- `faceswap/swap_and_bake/intermediate/reactor_swap_*`
- `faceswap/swap_and_bake/intermediate/bake_mask_*`
- `faceswap/swap_and_bake/final_*`

Visual-prompt hybrid fallback checkpoints:
- `faceswap/visual_prompt_hybrid/intermediate/target_head_mask_*`
- `faceswap/visual_prompt_hybrid/intermediate/generated_head_*`
- `faceswap/visual_prompt_hybrid/intermediate/reactor_bake_*`
- `faceswap/visual_prompt_hybrid/intermediate/inner_face_mask_*`
- `faceswap/visual_prompt_hybrid/intermediate/pre_skin_harmonize_*`
- `faceswap/visual_prompt_hybrid/intermediate/target_skin_mask_*`
- `faceswap/visual_prompt_hybrid/intermediate/target_skin_mask_refined_*`
- `faceswap/visual_prompt_hybrid/final_postprocess_*`
- `faceswap/visual_prompt_hybrid/final_*`

ComfyUI may cache unchanged branches. If a manual rerun appears not to write a fresh file, change an input, change `--filename-prefix` / `--intermediate-prefix`, or clear the server cache.

## 3) Minimum SimplePod spec

Minimum:
- NVIDIA GPU with 12 GB VRAM.
- 24 GB system RAM.
- 40 GB disk.
- Python 3.10+ ComfyUI image with CUDA/PyTorch.

Preferred for fewer memory problems:
- 16 GB+ VRAM.
- 32 GB system RAM.
- 80 GB+ disk.

## 4) Validation checklist

- Target image composition remains intact.
- Subject identity is transferred to target face.
- ReActor finds the target face automatically.
- `faceswap/final_*` is written under the ComfyUI output directory.

## 5) Known risk checks

- `ReActorFaceSwap` must appear in `/object_info`; installed files alone are not enough.
- `inswapper_128.onnx` must exist under `models/insightface/`.
- `GFPGANv1.4.pth` must exist under `models/facerestore_models/`.
- The first run is expected to reveal graph/schema issues, so record fixes in `docs/known_mistakes.md`.

## 6) Development cycle

When changing the pipeline, keep the API and UI workflows together:

1. Edit `scripts/build_faceswap_workflow.py`.
2. Regenerate both workflow files with `.venv/bin/python scripts/build_faceswap_workflow.py`.
3. Validate both JSON files with `python3 -m json.tool`.
4. Deploy with `.venv/bin/python scripts/simplepod.py deploy`.
5. Run `.venv/bin/python scripts/simplepod.py preflight`.
6. Smoke test API execution with `.venv/bin/python scripts/simplepod.py queue --wait 300`.
7. Load `faceswap_subject_on_character_ui` in the browser for human-in-the-loop checks.

Keep these roles distinct:
- `_api`: fast iteration and automation through `/prompt`.
- `_ui`: manual ComfyUI execution with visible nodes, links, previews, and saves.

## 7) UI Development on SimplePod

The Vite frontend and FastAPI backend must run on the SimplePod to have access to the local ComfyUI outputs.
**Do not run `npm build` or `npm run dev` locally.** The SimplePod is the default environment for UI testing and iteration.

### Setup and Deployment

1. Install Node.js on the SimplePod (if not already installed):
```bash
.venv/bin/python scripts/simplepod.py install-node
```

2. Sync the codebase to the SimplePod and build the UI remotely:
```bash
.venv/bin/python scripts/simplepod.py deploy-app
```

3. Start the FastAPI backend server on the SimplePod:
```bash
.venv/bin/python scripts/simplepod.py serve-app
```
The `serve-app` command will print instructions on how to access the UI.

### Active UI Development (Hot Reloading)

For active UI development, use VS Code Remote-SSH (or similar) to connect to the SimplePod instance.
1. Open `/app/faceswap_deploy` (or `/workspace/faceswap_deploy`) in the remote IDE.
2. In the remote terminal, start the Vite dev server:
```bash
npm run dev
```
3. In a second remote terminal, start the FastAPI backend:
```bash
python3 scripts/run_deploy_app.py
```
Forward both ports `5173` (Vite) and `8000` (FastAPI) to your local machine to test live changes.
