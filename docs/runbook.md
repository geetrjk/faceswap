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

Human-in-the-loop output checkpoints:
- `faceswap/intermediate/subject_input_*`
- `faceswap/intermediate/target_input_*`
- `faceswap/intermediate/plain_swap_*`
- `faceswap/final_*`

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
