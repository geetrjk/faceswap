# Runbook: ComfyUI Face Swap Workflow

## 1) Local run (recommended first)

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
git clone https://github.com/Gourieff/comfyui-reactor-node.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
```

### Required model files (minimum)

Place in ComfyUI model directories:
- swap model: `models/insightface/inswapper_128.onnx`
- face restore: `models/facerestore_models/GFPGANv1.4.pth`
- detector (Impact): `models/ultralytics/bbox/face_yolov8m.pt`
- your checkpoint + VAE in standard `models/checkpoints` and `models/vae`

### Use this repo workflow

From this repo root:

```bash
python scripts/build_faceswap_workflow.py
cp workflows/faceswap_subject_on_character_api.json /path/to/ComfyUI/user/default/workflows/
```

Then open ComfyUI, load the workflow, set:
- `subject.jpg` (human identity source)
- `target.png` (character target)

Outputs:
- `faceswap/final_*` (intermediate swap)
- `faceswap/refined_*` (post-detailer)

## 2) Optional SimplePod run

If you provide SimplePod env vars, you can use this same workflow remotely.

Suggested env variables:

```bash
export SIMPLEPOD_HOST=<ip-or-host>
export SIMPLEPOD_PORT=22
export SIMPLEPOD_USER=<user>
export SIMPLEPOD_KEY=~/.ssh/simplepod_key
```

Then sync workflow/assets and run ComfyUI on pod similarly.

## 3) Validation checklist

- Target image composition remains intact.
- Subject identity is transferred to target face.
- Face detector finds face automatically (no manual masks).
- Refined output improves skin/hairline/eye consistency.
