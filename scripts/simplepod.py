#!/usr/bin/env python3
"""Small SimplePod helper for the faceswap ComfyUI base pipeline."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path

try:
    import paramiko
except ModuleNotFoundError:  # pragma: no cover - friendly CLI failure
    paramiko = None


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
TEST_SUBJECTS_DIR = ROOT / "test_subjects"
WORKFLOW = ROOT / "workflows" / "faceswap_subject_on_character_api.json"
UI_WORKFLOW = ROOT / "workflows" / "faceswap_subject_on_character_ui.json"
INSTANTID_WORKFLOW = ROOT / "workflows" / "instantid_subject_pose_style_api.json"
INSTANTID_UI_WORKFLOW = ROOT / "workflows" / "instantid_subject_pose_style_ui.json"
INSTANTID_CROP_WORKFLOW = ROOT / "workflows" / "instantid_crop_stitch_experiment_api.json"
INSTANTID_CROP_UI_WORKFLOW = ROOT / "workflows" / "instantid_crop_stitch_experiment_ui.json"
SWAP_AND_BAKE_WORKFLOW = ROOT / "workflows" / "swap_and_bake_experiment_api.json"
SWAP_AND_BAKE_UI_WORKFLOW = ROOT / "workflows" / "swap_and_bake_experiment_ui.json"
VISUAL_PROMPT_HYBRID_WORKFLOW = ROOT / "workflows" / "visual_prompt_hybrid_experiment_api.json"
VISUAL_PROMPT_HYBRID_UI_WORKFLOW = ROOT / "workflows" / "visual_prompt_hybrid_experiment_ui.json"
REMOTE_SKIN_TONE_POSTPROCESS = ROOT / "scripts" / "remote_skin_tone_postprocess.py"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_INPUTS = [
    ROOT / "subject_5 year curly.webp",
]
ROOT_IMAGE_INPUTS = sorted(
    path
    for path in ROOT.iterdir()
    if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
)
INPUTS: list[Path] = []
for path in [*DEFAULT_INPUTS, *ROOT_IMAGE_INPUTS]:
    if path.exists() and path not in INPUTS:
        INPUTS.append(path)
VISUAL_PROMPT_INPUTS = [
    *INPUTS,
    *sorted(path for path in TEST_SUBJECTS_DIR.iterdir() if path.is_file() and not path.name.startswith(".")),
]
REQUIRED_NODES = [
    "ReActorFaceSwap",
]
REQUIRED_MODEL_PATHS = [
    "models/insightface/inswapper_128.onnx",
    "models/facerestore_models/GFPGANv1.4.pth",
]
INSTANTID_REQUIRED_NODES = [
    "ApplyInstantIDAdvanced",
    "FaceAnalysisModels",
    "FaceKeypointsPreprocessor",
    "FaceSegmentation",
    "ImageCompositeMasked",
    "InstantIDFaceAnalysis",
    "InstantIDModelLoader",
    "MaskToImage",
    "VAEEncodeForInpaint",
]
INSTANTID_CROP_REQUIRED_NODES = [
    *INSTANTID_REQUIRED_NODES,
    "Canny",
    "ControlNetApplyAdvanced",
    "CropMask",
    "GrowMask",
    "ImageCrop",
]
INSTANTID_REQUIRED_MODEL_PATHS = [
    "models/checkpoints/sd_xl_base_1.0.safetensors",
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors",
    "models/instantid/ip-adapter.bin",
    "models/controlnet/instantid_controlnet.safetensors",
    "models/insightface/models/antelopev2/1k3d68.onnx",
    "models/insightface/models/antelopev2/2d106det.onnx",
    "models/insightface/models/antelopev2/genderage.onnx",
    "models/insightface/models/antelopev2/glintr100.onnx",
    "models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
    "models/insightface/models/buffalo_l/1k3d68.onnx",
    "models/insightface/models/buffalo_l/2d106det.onnx",
    "models/insightface/models/buffalo_l/det_10g.onnx",
    "models/insightface/models/buffalo_l/genderage.onnx",
    "models/insightface/models/buffalo_l/w600k_r50.onnx",
]
INSTANTID_CROP_REQUIRED_MODEL_PATHS = [
    *INSTANTID_REQUIRED_MODEL_PATHS,
    "models/controlnet/controlnet-canny-sdxl-1.0-small.safetensors",
]
INSTANTID_CUSTOM_NODES = {
    "ComfyUI_InstantID": "https://github.com/cubiq/ComfyUI_InstantID.git",
    "ComfyUI_FaceAnalysis": "https://github.com/cubiq/ComfyUI_FaceAnalysis.git",
}
INSTANTID_CUSTOM_NODE_ARCHIVES = {
    "ComfyUI_InstantID": "https://codeload.github.com/cubiq/ComfyUI_InstantID/zip/refs/heads/main",
    "ComfyUI_FaceAnalysis": "https://codeload.github.com/cubiq/ComfyUI_FaceAnalysis/zip/refs/heads/main",
}
REACTOR_CUSTOM_NODES = {
    "ComfyUI-ReActor": "https://github.com/Gourieff/ComfyUI-ReActor.git",
}
REACTOR_CUSTOM_NODE_ARCHIVES = {
    "ComfyUI-ReActor": "https://codeload.github.com/Gourieff/ComfyUI-ReActor/zip/refs/heads/main",
}
REACTOR_MODEL_URLS = {
    "models/insightface/inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "models/facerestore_models/GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}
REACTOR_MODEL_MIN_BYTES = {
    "models/insightface/inswapper_128.onnx": 500_000_000,
    "models/facerestore_models/GFPGANv1.4.pth": 300_000_000,
}
INSTANTID_MODEL_URLS = {
    "models/checkpoints/sd_xl_base_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors": "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_inpainting_0.1.safetensors",
    "models/instantid/ip-adapter.bin": "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin",
    "models/controlnet/instantid_controlnet.safetensors": "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors",
    "models/controlnet/controlnet-canny-sdxl-1.0-small.safetensors": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.safetensors",
}
INSTANTID_MODEL_MIN_BYTES = {
    "models/checkpoints/sd_xl_base_1.0.safetensors": 6_000_000_000,
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors": 6_000_000_000,
    "models/instantid/ip-adapter.bin": 1_000_000_000,
    "models/controlnet/instantid_controlnet.safetensors": 2_000_000_000,
    "models/controlnet/controlnet-canny-sdxl-1.0-small.safetensors": 600_000_000,
}
INSTANTID_ARCHIVE_URLS = {
    "antelopev2": (
        "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip",
        "models/insightface/models/antelopev2",
        "1k3d68.onnx",
    ),
    "buffalo_l": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "models/insightface/models/buffalo_l",
        "det_10g.onnx",
    ),
}
VISUAL_PROMPT_REQUIRED_MODEL_PATHS = [
    "models/checkpoints/sd_xl_base_1.0.safetensors",
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors",
    "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors",
    "models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors",
    "models/sams/sam_vit_b_01ec64.pth",
    "models/insightface/models/antelopev2/1k3d68.onnx",
    "models/insightface/models/antelopev2/2d106det.onnx",
    "models/insightface/models/antelopev2/genderage.onnx",
    "models/insightface/models/antelopev2/glintr100.onnx",
    "models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
]
VISUAL_PROMPT_REQUIRED_NODES = [
    "ApplyPulidAdvanced",
    "CLIPSeg",
    "FaceAnalysisModels",
    "FaceSegmentation",
    "ImageCompositeMasked",
    "IPAdapterAdvanced",
    "IPAdapterUnifiedLoader",
    "PulidEvaClipLoader",
    "PulidInsightFaceLoader",
    "PulidModelLoader",
    "ReActorFaceSwap",
    "VAEEncodeForInpaint",
]
VISUAL_PROMPT_CUSTOM_NODES = {
    "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
    "PuLID_ComfyUI": "https://github.com/cubiq/PuLID_ComfyUI.git",
    "ComfyUI-Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
    "ComfyUI-CLIPSeg": "https://github.com/time-river/ComfyUI-CLIPSeg.git",
    "ComfyUI_FaceAnalysis": "https://github.com/cubiq/ComfyUI_FaceAnalysis.git",
}
VISUAL_PROMPT_CUSTOM_NODE_ARCHIVES = {
    "ComfyUI_IPAdapter_plus": "https://codeload.github.com/cubiq/ComfyUI_IPAdapter_plus/zip/refs/heads/main",
    "PuLID_ComfyUI": "https://codeload.github.com/cubiq/PuLID_ComfyUI/zip/refs/heads/main",
    "ComfyUI-Impact-Pack": "https://codeload.github.com/ltdrdata/ComfyUI-Impact-Pack/zip/refs/heads/Main",
    "ComfyUI-CLIPSeg": "https://codeload.github.com/time-river/ComfyUI-CLIPSeg/zip/refs/heads/main",
    "ComfyUI_FaceAnalysis": "https://codeload.github.com/cubiq/ComfyUI_FaceAnalysis/zip/refs/heads/main",
}
VISUAL_PROMPT_MODEL_URLS = {
    "models/checkpoints/sd_xl_base_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors": "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_inpainting_0.1.safetensors",
    "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "https://huggingface.co/fofr/comfyui/resolve/main/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
    "models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors": "https://huggingface.co/8clabs/models-moved/resolve/main/pulid/ip-adapter_pulid_sdxl_fp16.safetensors",
    "models/sams/sam_vit_b_01ec64.pth": "https://huggingface.co/scenario-labs/sam_vit/resolve/main/sam_vit_b_01ec64.pth",
}
VISUAL_PROMPT_MODEL_MIN_BYTES = {
    "models/checkpoints/sd_xl_base_1.0.safetensors": 6_000_000_000,
    "models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors": 6_000_000_000,
    "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": 2_000_000_000,
    "models/ipadapter/ip-adapter_sdxl_vit-h.safetensors": 650_000_000,
    "models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors": 700_000_000,
    "models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors": 700_000_000,
    "models/sams/sam_vit_b_01ec64.pth": 300_000_000,
}


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return {**values, **os.environ}


def require_paramiko() -> None:
    if paramiko is None:
        raise SystemExit("Missing dependency: run `.venv/bin/python -m pip install -r requirements.txt`.")


def connect():
    require_paramiko()
    env = load_env()
    host = env.get("SIMPLEPOD_SSH_HOST")
    if not host:
        raise SystemExit("Missing SIMPLEPOD_SSH_HOST in .env.")
    port = int(env.get("SIMPLEPOD_SSH_PORT", "22"))
    user = env.get("SIMPLEPOD_SSH_USER", "root")
    password = env.get("SIMPLEPOD_PASSWORD")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=user, password=password, timeout=20)
    except Exception as exc:
        raise SystemExit(
            "Could not connect to SimplePod over SSH. "
            "Check that the pod is running and refresh SIMPLEPOD_SSH_HOST/SIMPLEPOD_SSH_PORT in .env. "
            f"Paramiko error: {exc.__class__.__name__}"
        ) from exc
    return client


def run_remote(client, command: str, *, check: bool = True, stdin_data: str | None = None) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    if stdin_data is not None:
        stdin.write(stdin_data)
        stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if check and code != 0:
        raise RuntimeError(f"Remote command failed ({code}): {command}\n{err.strip()}")
    return code, out, err


def run_remote_stream(client, command: str) -> int:
    stdin, stdout, _stderr = client.exec_command(command)
    stdin.channel.shutdown_write()
    channel = stdout.channel
    while not channel.exit_status_ready():
        while channel.recv_ready():
            sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
            sys.stdout.flush()
        while channel.recv_stderr_ready():
            sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
            sys.stderr.flush()
        time.sleep(0.2)

    while channel.recv_ready():
        sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
        sys.stdout.flush()
    while channel.recv_stderr_ready():
        sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
        sys.stderr.flush()
    return channel.recv_exit_status()


def find_comfy_root(client) -> str:
    command = "for d in /app/ComfyUI /workspace/ComfyUI; do [ -d \"$d\" ] && echo \"$d\" && exit 0; done; exit 1"
    code, out, _ = run_remote(client, command, check=False)
    if code != 0 or not out.strip():
        raise RuntimeError("Could not find ComfyUI at /app/ComfyUI or /workspace/ComfyUI.")
    return out.strip().splitlines()[0]


def profile(_args) -> None:
    with connect() as client:
        checks = [
            ("GPU", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"),
            ("Disk", "df -h / /app /workspace 2>/dev/null || df -h /"),
            ("Python", "python --version || python3 --version"),
            ("pip", "which pip || which pip3 || true"),
            ("ComfyUI", "for d in /app/ComfyUI /workspace/ComfyUI; do [ -d \"$d\" ] && echo \"$d\"; done"),
        ]
        for label, command in checks:
            _, out, err = run_remote(client, command, check=False)
            print(f"== {label} ==")
            print((out or err).strip() or "not found")


def deploy(_args) -> None:
    for path in [WORKFLOW, UI_WORKFLOW, *INPUTS]:
        if not path.exists():
            raise SystemExit(f"Missing local file: {path}")

    with connect() as client:
        comfy_root = find_comfy_root(client)
        workflow_dir = posixpath.join(comfy_root, "user/default/workflows/faceswap")
        input_dir = posixpath.join(comfy_root, "input")
        run_remote(client, f"mkdir -p {shlex.quote(workflow_dir)} {shlex.quote(input_dir)}")

        sftp = client.open_sftp()
        try:
            remote_workflow = posixpath.join(workflow_dir, WORKFLOW.name)
            print(f"Uploading {WORKFLOW.name} -> {remote_workflow}")
            sftp.put(str(WORKFLOW), remote_workflow)
            remote_ui_workflow = posixpath.join(workflow_dir, UI_WORKFLOW.name)
            print(f"Uploading {UI_WORKFLOW.name} -> {remote_ui_workflow}")
            sftp.put(str(UI_WORKFLOW), remote_ui_workflow)
            for path in INPUTS:
                remote_path = posixpath.join(input_dir, path.name)
                print(f"Uploading {path.name} -> {remote_path}")
                sftp.put(str(path), remote_path)
        finally:
            sftp.close()

        print(f"Deployed to {workflow_dir}")


def deploy_instantid(_args) -> None:
    for path in [INSTANTID_WORKFLOW, INSTANTID_UI_WORKFLOW, *INPUTS]:
        if not path.exists():
            raise SystemExit(f"Missing local file: {path}")

    with connect() as client:
        comfy_root = find_comfy_root(client)
        workflow_dir = posixpath.join(comfy_root, "user/default/workflows/faceswap")
        input_dir = posixpath.join(comfy_root, "input")
        run_remote(client, f"mkdir -p {shlex.quote(workflow_dir)} {shlex.quote(input_dir)}")

        sftp = client.open_sftp()
        try:
            for path in [INSTANTID_WORKFLOW, INSTANTID_UI_WORKFLOW]:
                remote_workflow = posixpath.join(workflow_dir, path.name)
                print(f"Uploading {path.name} -> {remote_workflow}")
                sftp.put(str(path), remote_workflow)
            for path in INPUTS:
                remote_path = posixpath.join(input_dir, path.name)
                print(f"Uploading {path.name} -> {remote_path}")
                sftp.put(str(path), remote_path)
        finally:
            sftp.close()

        print(f"Deployed InstantID experiment to {workflow_dir}")


def deploy_instantid_crop(_args) -> None:
    for path in [INSTANTID_CROP_WORKFLOW, INSTANTID_CROP_UI_WORKFLOW, *INPUTS]:
        if not path.exists():
            raise SystemExit(f"Missing local file: {path}")

    with connect() as client:
        comfy_root = find_comfy_root(client)
        workflow_dir = posixpath.join(comfy_root, "user/default/workflows/faceswap")
        input_dir = posixpath.join(comfy_root, "input")
        run_remote(client, f"mkdir -p {shlex.quote(workflow_dir)} {shlex.quote(input_dir)}")

        sftp = client.open_sftp()
        try:
            for path in [INSTANTID_CROP_WORKFLOW, INSTANTID_CROP_UI_WORKFLOW]:
                remote_workflow = posixpath.join(workflow_dir, path.name)
                print(f"Uploading {path.name} -> {remote_workflow}")
                sftp.put(str(path), remote_workflow)
            for path in INPUTS:
                remote_path = posixpath.join(input_dir, path.name)
                print(f"Uploading {path.name} -> {remote_path}")
                sftp.put(str(path), remote_path)
        finally:
            sftp.close()

        print(f"Deployed InstantID crop-stitch experiment to {workflow_dir}")


def deploy_swap_and_bake(_args) -> None:
    for path in [SWAP_AND_BAKE_WORKFLOW, SWAP_AND_BAKE_UI_WORKFLOW, *INPUTS]:
        if not path.exists():
            raise SystemExit(f"Missing local file: {path}")

    with connect() as client:
        comfy_root = find_comfy_root(client)
        workflow_dir = posixpath.join(comfy_root, "user/default/workflows/faceswap")
        input_dir = posixpath.join(comfy_root, "input")
        run_remote(client, f"mkdir -p {shlex.quote(workflow_dir)} {shlex.quote(input_dir)}")

        sftp = client.open_sftp()
        try:
            for path in [SWAP_AND_BAKE_WORKFLOW, SWAP_AND_BAKE_UI_WORKFLOW]:
                remote_workflow = posixpath.join(workflow_dir, path.name)
                print(f"Uploading {path.name} -> {remote_workflow}")
                sftp.put(str(path), remote_workflow)
            for path in INPUTS:
                remote_path = posixpath.join(input_dir, path.name)
                print(f"Uploading {path.name} -> {remote_path}")
                sftp.put(str(path), remote_path)
        finally:
            sftp.close()

        print(f"Deployed swap-and-bake experiment to {workflow_dir}")


def deploy_visual_prompt_hybrid(_args) -> None:
    for path in [VISUAL_PROMPT_HYBRID_WORKFLOW, VISUAL_PROMPT_HYBRID_UI_WORKFLOW, *VISUAL_PROMPT_INPUTS]:
        if not path.exists():
            raise SystemExit(f"Missing local file: {path}")

    with connect() as client:
        comfy_root = find_comfy_root(client)
        workflow_dir = posixpath.join(comfy_root, "user/default/workflows/faceswap")
        input_dir = posixpath.join(comfy_root, "input")
        run_remote(client, f"mkdir -p {shlex.quote(workflow_dir)} {shlex.quote(input_dir)}")

        sftp = client.open_sftp()
        try:
            for path in [VISUAL_PROMPT_HYBRID_WORKFLOW, VISUAL_PROMPT_HYBRID_UI_WORKFLOW]:
                remote_workflow = posixpath.join(workflow_dir, path.name)
                print(f"Uploading {path.name} -> {remote_workflow}")
                sftp.put(str(path), remote_workflow)
            for path in VISUAL_PROMPT_INPUTS:
                remote_path = posixpath.join(input_dir, path.name)
                print(f"Uploading {path.name} -> {remote_path}")
                sftp.put(str(path), remote_path)
        finally:
            sftp.close()

        print(f"Deployed visual-prompt hybrid experiment to {workflow_dir}")


def preflight_visual_prompt(_args) -> None:
    port = getattr(_args, "port", 8188)
    with connect() as client:
        comfy_root = find_comfy_root(client)
        print(f"ComfyUI root: {comfy_root}")

        model_checks = " ; ".join(
            f"[ -f {shlex.quote(posixpath.join(comfy_root, rel))} ] && echo OK:{shlex.quote(rel)} || echo MISSING:{shlex.quote(rel)}"
            for rel in VISUAL_PROMPT_REQUIRED_MODEL_PATHS
        )
        _, model_out, _ = run_remote(client, model_checks, check=False)
        print("== Visual prompt model files ==")
        print(model_out.strip())

        object_info_cmd = (
            "python3 - <<'PY'\n"
            "import json, urllib.request\n"
            "token_path = '/app/ComfyUI/login/PASSWORD'\n"
            "token = ''\n"
            "try:\n"
            "    token = open(token_path, encoding='utf-8').readline().strip()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            f"req = urllib.request.Request('http://127.0.0.1:{port}/object_info')\n"
            "if token:\n"
            "    req.add_header('Authorization', 'Bearer ' + token)\n"
            "data = json.load(urllib.request.urlopen(req, timeout=10))\n"
            f"wanted = {VISUAL_PROMPT_REQUIRED_NODES!r}\n"
            "for name in wanted:\n"
            "    print(('OK:' if name in data else 'MISSING:') + name)\n"
            "PY"
        )
        code, node_out, node_err = run_remote(client, object_info_cmd, check=False)
        print("== Visual prompt live node registry ==")
        print((node_out or node_err).strip())
        if code != 0:
            print(f"Could not query local ComfyUI /object_info. Is ComfyUI running on port {port}?", file=sys.stderr)


def init_auth(_args) -> None:
    env = load_env()
    username = env.get("SIMPLEPOD_SSH_USER", "root")
    password = env.get("SIMPLEPOD_PASSWORD")
    if not password:
        raise SystemExit("Missing SIMPLEPOD_PASSWORD in .env.")

    script = r"""
import os
import sys
import bcrypt

password, username = sys.stdin.read().split("\n", 1)
username = username.strip() or "root"
path = "/app/ComfyUI/login/PASSWORD"
os.makedirs(os.path.dirname(path), exist_ok=True)
if os.path.exists(path) and os.path.getsize(path) > 0:
    print("Authenticator password file already exists.")
    raise SystemExit(0)

hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
with open(path, "wb") as file:
    file.write(hashed + b"\n" + username.encode("utf-8"))
os.chmod(path, 0o600)
print("Authenticator password file initialized. Restart ComfyUI for API token use.")
"""
    with connect() as client:
        run_remote(client, f"python3 -c {shlex.quote(script)}", stdin_data=f"{password}\n{username}\n")


def preflight(_args) -> None:
    port = getattr(_args, "port", 8188)
    with connect() as client:
        comfy_root = find_comfy_root(client)
        print(f"ComfyUI root: {comfy_root}")

        model_checks = " ; ".join(
            f"[ -f {shlex.quote(posixpath.join(comfy_root, rel))} ] && echo OK:{shlex.quote(rel)} || echo MISSING:{shlex.quote(rel)}"
            for rel in REQUIRED_MODEL_PATHS
        )
        _, model_out, _ = run_remote(client, model_checks, check=False)
        print("== Model files ==")
        print(model_out.strip())

        object_info_cmd = (
            "python3 - <<'PY'\n"
            "import json, urllib.request\n"
            "token_path = '/app/ComfyUI/login/PASSWORD'\n"
            "token = ''\n"
            "try:\n"
            "    token = open(token_path, encoding='utf-8').readline().strip()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            f"req = urllib.request.Request('http://127.0.0.1:{port}/object_info')\n"
            "if token:\n"
            "    req.add_header('Authorization', 'Bearer ' + token)\n"
            "data = json.load(urllib.request.urlopen(req, timeout=10))\n"
            f"wanted = {REQUIRED_NODES!r}\n"
            "for name in wanted:\n"
            "    print(('OK:' if name in data else 'MISSING:') + name)\n"
            "PY"
        )
        code, node_out, node_err = run_remote(client, object_info_cmd, check=False)
        print("== Live node registry ==")
        print((node_out or node_err).strip())
        if code != 0:
            print("Could not query local ComfyUI /object_info. Is ComfyUI running on port 8188?", file=sys.stderr)


def preflight_instantid(_args) -> None:
    port = getattr(_args, "port", 8188)
    crop_stitch = getattr(_args, "crop_stitch", False)
    required_nodes = INSTANTID_CROP_REQUIRED_NODES if crop_stitch else INSTANTID_REQUIRED_NODES
    required_model_paths = INSTANTID_CROP_REQUIRED_MODEL_PATHS if crop_stitch else INSTANTID_REQUIRED_MODEL_PATHS
    with connect() as client:
        comfy_root = find_comfy_root(client)
        print(f"ComfyUI root: {comfy_root}")

        model_checks = " ; ".join(
            f"[ -f {shlex.quote(posixpath.join(comfy_root, rel))} ] && echo OK:{shlex.quote(rel)} || echo MISSING:{shlex.quote(rel)}"
            for rel in required_model_paths
        )
        _, model_out, _ = run_remote(client, model_checks, check=False)
        print("== InstantID model files ==")
        print(model_out.strip())

        object_info_cmd = (
            "python3 - <<'PY'\n"
            "import json, urllib.request\n"
            "token_path = '/app/ComfyUI/login/PASSWORD'\n"
            "token = ''\n"
            "try:\n"
            "    token = open(token_path, encoding='utf-8').readline().strip()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            f"req = urllib.request.Request('http://127.0.0.1:{port}/object_info')\n"
            "if token:\n"
            "    req.add_header('Authorization', 'Bearer ' + token)\n"
            "data = json.load(urllib.request.urlopen(req, timeout=10))\n"
            f"wanted = {required_nodes!r}\n"
            "for name in wanted:\n"
            "    print(('OK:' if name in data else 'MISSING:') + name)\n"
            "PY"
        )
        code, node_out, node_err = run_remote(client, object_info_cmd, check=False)
        print("== InstantID live node registry ==")
        print((node_out or node_err).strip())
        if code != 0:
            print(f"Could not query local ComfyUI /object_info. Is ComfyUI running on port {port}?", file=sys.stderr)


def install_reactor(_args) -> None:
    node_lines = "\n".join(
        f"ensure_node {shlex.quote(name)} {shlex.quote(url)} {shlex.quote(REACTOR_CUSTOM_NODE_ARCHIVES[name])}"
        for name, url in REACTOR_CUSTOM_NODES.items()
    )
    model_lines = "\n".join(
        f"ensure_file {shlex.quote(rel)} {shlex.quote(url)} {REACTOR_MODEL_MIN_BYTES[rel]}"
        for rel, url in REACTOR_MODEL_URLS.items()
    )
    script = f"""
set -euo pipefail
cd "$COMFY_ROOT"

ensure_node() {{
  name="$1"
  git_url="$2"
  archive_url="$3"
  path="custom_nodes/$name"
  if [ -d "$path/.git" ]; then
    echo "Updating $name"
    timeout 30 git -C "$path" pull --ff-only || true
  elif [ -d "$path" ]; then
    echo "Keeping existing non-git custom node $name"
  else
    echo "Cloning $name"
    if ! timeout 30 git clone --depth 1 "$git_url" "$path"; then
      echo "git clone failed for $name; downloading archive"
      archive="/tmp/$name.zip"
      tmp_dir="/tmp/$name-unpack"
      rm -rf "$tmp_dir" "$archive"
      mkdir -p "$tmp_dir"
      curl -L --fail --connect-timeout 15 --max-time 120 -o "$archive" "$archive_url"
      python3 -m zipfile -e "$archive" "$tmp_dir"
      extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      [ -n "$extracted" ]
      rm -rf "$path"
      mv "$extracted" "$path"
    fi
  fi
}}

ensure_requirements() {{
  path="$1"
  if [ -f "$path/requirements.txt" ]; then
    echo "Installing requirements for $path"
    python3 -m pip install --disable-pip-version-check --root-user-action=ignore -q -r "$path/requirements.txt"
  fi
}}

ensure_file() {{
  rel="$1"
  url="$2"
  min_bytes="$3"
  mkdir -p "$(dirname "$rel")"
  size=0
  if [ -f "$rel" ]; then
    size="$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$rel")"
  fi
  if [ "$size" -ge "$min_bytes" ]; then
    echo "OK:$rel ($size bytes)"
  else
    echo "Downloading/resuming $rel ($size bytes; need at least $min_bytes)"
    curl -L --fail --connect-timeout 15 --retry 5 --retry-delay 5 -C - -o "$rel" "$url"
  fi
}}

mkdir -p custom_nodes models/insightface models/facerestore_models

{node_lines}

ensure_requirements custom_nodes/ComfyUI-ReActor
python3 custom_nodes/ComfyUI-ReActor/install.py || true

{model_lines}

python3 - <<'PY'
from pathlib import Path

node_dir = Path("custom_nodes/ComfyUI-ReActor")
print("FILE_CHECK:custom_nodes/ComfyUI-ReActor")
if not node_dir.exists():
    raise SystemExit("missing custom_nodes/ComfyUI-ReActor")
text = "\\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in node_dir.glob("*.py"))
if "ReActorFaceSwap" not in text:
    raise SystemExit("ReActorFaceSwap not found in installed files")
print("OK:ReActorFaceSwap")
PY
"""
    with connect() as client:
        comfy_root = find_comfy_root(client)
        command = f"COMFY_ROOT={shlex.quote(comfy_root)} bash -lc {shlex.quote(script)}"
        code = run_remote_stream(client, command)
        if code != 0:
            raise SystemExit(code)


def install_instantid(_args) -> None:
    node_lines = "\n".join(
        f"ensure_node {shlex.quote(name)} {shlex.quote(url)} {shlex.quote(INSTANTID_CUSTOM_NODE_ARCHIVES[name])}"
        for name, url in INSTANTID_CUSTOM_NODES.items()
    )
    model_lines = "\n".join(
        f"ensure_file {shlex.quote(rel)} {shlex.quote(url)} {INSTANTID_MODEL_MIN_BYTES[rel]}"
        for rel, url in INSTANTID_MODEL_URLS.items()
    )
    archive_lines = "\n".join(
        f"ensure_archive {shlex.quote(name)} {shlex.quote(url)} {shlex.quote(rel_dir)} {shlex.quote(marker)}"
        for name, (url, rel_dir, marker) in INSTANTID_ARCHIVE_URLS.items()
    )
    script = f"""
set -euo pipefail
cd "$COMFY_ROOT"

ensure_node() {{
  name="$1"
  git_url="$2"
  archive_url="$3"
  path="custom_nodes/$name"
  if [ -d "$path/.git" ]; then
    echo "Updating $name"
    timeout 30 git -C "$path" pull --ff-only || true
  elif [ -d "$path" ]; then
    echo "Keeping existing non-git custom node $name"
  else
    echo "Cloning $name"
    if ! timeout 30 git clone --depth 1 "$git_url" "$path"; then
      echo "git clone failed for $name; downloading archive"
      archive="/tmp/$name.zip"
      tmp_dir="/tmp/$name-unpack"
      rm -rf "$tmp_dir" "$archive"
      mkdir -p "$tmp_dir"
      curl -L --fail --connect-timeout 15 --max-time 120 -o "$archive" "$archive_url"
      python3 -m zipfile -e "$archive" "$tmp_dir"
      extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      [ -n "$extracted" ]
      rm -rf "$path"
      mv "$extracted" "$path"
    fi
  fi
}}

ensure_requirements() {{
  path="$1"
  if [ -f "$path/requirements.txt" ]; then
    echo "Installing requirements for $path"
    python3 -m pip install --disable-pip-version-check --root-user-action=ignore -q -r "$path/requirements.txt"
  fi
}}

ensure_python_module() {{
  module_name="$1"
  package_name="$2"
  if python3 -c "import $module_name" >/dev/null 2>&1; then
    echo "OK:python_module:$module_name"
  else
    echo "Installing python package $package_name for module $module_name"
    python3 -m pip install --disable-pip-version-check --root-user-action=ignore -q "$package_name"
  fi
}}

ensure_file() {{
  rel="$1"
  url="$2"
  min_bytes="$3"
  mkdir -p "$(dirname "$rel")"
  size=0
  if [ -f "$rel" ]; then
    size="$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$rel")"
  fi
  if [ "$size" -ge "$min_bytes" ]; then
    echo "OK:$rel ($size bytes)"
  else
    echo "Downloading/resuming $rel ($size bytes; need at least $min_bytes)"
    curl -L --fail --connect-timeout 15 --retry 5 --retry-delay 5 -C - -o "$rel" "$url"
  fi
}}

ensure_archive() {{
  name="$1"
  url="$2"
  rel_dir="$3"
  marker="$4"
  archive="/tmp/$name.zip"
  if [ -s "$rel_dir/$marker" ]; then
    echo "OK:$rel_dir/$marker"
  else
    echo "Downloading and extracting $name"
    rm -rf "$rel_dir"
    mkdir -p "$rel_dir"
    curl -L --fail --connect-timeout 15 --retry 5 --retry-delay 5 -o "$archive" "$url"
    python3 -m zipfile -e "$archive" "$rel_dir"
    if [ ! -s "$rel_dir/$marker" ]; then
      nested_marker="$(find "$rel_dir" -mindepth 2 -maxdepth 2 -type f -name "$marker" | head -n 1)"
      if [ -n "$nested_marker" ]; then
        nested_dir="$(dirname "$nested_marker")"
        find "$nested_dir" -mindepth 1 -maxdepth 1 -exec mv -t "$rel_dir" {{}} +
        rmdir "$nested_dir" || true
      fi
    fi
    [ -s "$rel_dir/$marker" ]
  fi
}}

mkdir -p custom_nodes models/checkpoints models/instantid models/controlnet models/insightface/models

{node_lines}

ensure_requirements custom_nodes/ComfyUI_InstantID
ensure_requirements custom_nodes/ComfyUI_FaceAnalysis

{model_lines}
{archive_lines}

python3 - <<'PY'
import traceback
for name in ["custom_nodes.ComfyUI_InstantID", "custom_nodes.ComfyUI_FaceAnalysis"]:
    print("IMPORT_CHECK:" + name)
    try:
        mod = __import__(name, fromlist=["NODE_CLASS_MAPPINGS"])
        print(",".join(sorted(mod.NODE_CLASS_MAPPINGS)))
    except Exception:
        traceback.print_exc()
        raise
PY
"""
    with connect() as client:
        comfy_root = find_comfy_root(client)
        command = f"COMFY_ROOT={shlex.quote(comfy_root)} bash -lc {shlex.quote(script)}"
        code = run_remote_stream(client, command)
        if code != 0:
            raise SystemExit(code)


def install_visual_prompt_stack(_args) -> None:
    node_lines = "\n".join(
        f"ensure_node {shlex.quote(name)} {shlex.quote(url)} {shlex.quote(VISUAL_PROMPT_CUSTOM_NODE_ARCHIVES[name])}"
        for name, url in VISUAL_PROMPT_CUSTOM_NODES.items()
    )
    model_lines = "\n".join(
        f"ensure_file {shlex.quote(rel)} {shlex.quote(url)} {VISUAL_PROMPT_MODEL_MIN_BYTES[rel]}"
        for rel, url in VISUAL_PROMPT_MODEL_URLS.items()
    )
    archive_lines = "\n".join(
        f"ensure_archive {shlex.quote(name)} {shlex.quote(url)} {shlex.quote(rel_dir)} {shlex.quote(marker)}"
        for name, (url, rel_dir, marker) in INSTANTID_ARCHIVE_URLS.items()
    )
    script = f"""
set -euo pipefail
cd "$COMFY_ROOT"

ensure_node() {{
  name="$1"
  git_url="$2"
  archive_url="$3"
  path="custom_nodes/$name"
  if [ -d "$path/.git" ]; then
    echo "Updating $name"
    timeout 30 git -C "$path" pull --ff-only || true
  elif [ -d "$path" ]; then
    echo "Keeping existing non-git custom node $name"
  else
    echo "Cloning $name"
    if ! timeout 30 git clone --depth 1 "$git_url" "$path"; then
      echo "git clone failed for $name; downloading archive"
      archive="/tmp/$name.zip"
      tmp_dir="/tmp/$name-unpack"
      rm -rf "$tmp_dir" "$archive"
      mkdir -p "$tmp_dir"
      curl -L --fail --connect-timeout 15 --max-time 120 -o "$archive" "$archive_url"
      python3 -m zipfile -e "$archive" "$tmp_dir"
      extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
      [ -n "$extracted" ]
      rm -rf "$path"
      mv "$extracted" "$path"
    fi
  fi
}}

ensure_requirements() {{
  path="$1"
  if [ -f "$path/requirements.txt" ]; then
    echo "Installing requirements for $path"
    python3 -m pip install --disable-pip-version-check --root-user-action=ignore -q -r "$path/requirements.txt"
  fi
}}

ensure_python_module() {{
  module_name="$1"
  package_name="$2"
  if python3 -c "import $module_name" >/dev/null 2>&1; then
    echo "OK:python_module:$module_name"
  else
    echo "Installing python package $package_name for module $module_name"
    python3 -m pip install --disable-pip-version-check --root-user-action=ignore -q "$package_name"
  fi
}}

ensure_file() {{
  rel="$1"
  url="$2"
  min_bytes="$3"
  mkdir -p "$(dirname "$rel")"
  size=0
  if [ -f "$rel" ]; then
    size="$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$rel")"
  fi
  if [ "$size" -ge "$min_bytes" ]; then
    echo "OK:$rel ($size bytes)"
  else
    echo "Downloading/resuming $rel ($size bytes; need at least $min_bytes)"
    curl -L --fail --connect-timeout 15 --retry 5 --retry-delay 5 -C - -o "$rel" "$url"
  fi
}}

ensure_archive() {{
  name="$1"
  url="$2"
  rel_dir="$3"
  marker="$4"
  archive="/tmp/$name.zip"
  if [ -s "$rel_dir/$marker" ]; then
    echo "OK:$rel_dir/$marker"
  else
    echo "Downloading and extracting $name"
    rm -rf "$rel_dir"
    mkdir -p "$rel_dir"
    curl -L --fail --connect-timeout 15 --retry 5 --retry-delay 5 -o "$archive" "$url"
    python3 -m zipfile -e "$archive" "$rel_dir"
    if [ ! -s "$rel_dir/$marker" ]; then
      nested_marker="$(find "$rel_dir" -mindepth 2 -maxdepth 2 -type f -name "$marker" | head -n 1)"
      if [ -n "$nested_marker" ]; then
        nested_dir="$(dirname "$nested_marker")"
        find "$nested_dir" -mindepth 1 -maxdepth 1 -exec mv -t "$rel_dir" {{}} +
        rmdir "$nested_dir" || true
      fi
    fi
    [ -s "$rel_dir/$marker" ]
  fi
}}

mkdir -p custom_nodes models/checkpoints models/clip_vision models/ipadapter models/pulid models/sams models/insightface/models

{node_lines}

ensure_requirements custom_nodes/ComfyUI_IPAdapter_plus
ensure_requirements custom_nodes/PuLID_ComfyUI
ensure_requirements custom_nodes/ComfyUI-Impact-Pack
ensure_requirements custom_nodes/ComfyUI_FaceAnalysis
ensure_python_module transformers transformers
ensure_python_module scipy scipy
ensure_python_module matplotlib matplotlib
ensure_python_module cv2 opencv-python
ensure_python_module PIL Pillow
if [ ! -f custom_nodes/ComfyUI-CLIPSeg/__init__.py ] && [ -f custom_nodes/ComfyUI-CLIPSeg/custom_nodes/clipseg.py ]; then
  echo "Creating ComfyUI-CLIPSeg loader shim"
  cat > custom_nodes/ComfyUI-CLIPSeg/__init__.py <<'PY'
from .custom_nodes.clipseg import CLIPSeg

NODE_CLASS_MAPPINGS = {{
    "CLIPSeg": CLIPSeg,
}}

NODE_DISPLAY_NAME_MAPPINGS = {{
    "CLIPSeg": "CLIPSeg",
}}
PY
fi

{model_lines}
{archive_lines}

python3 - <<'PY'
from pathlib import Path

required = [
    Path("custom_nodes/ComfyUI_IPAdapter_plus"),
    Path("custom_nodes/PuLID_ComfyUI"),
    Path("custom_nodes/ComfyUI-Impact-Pack"),
    Path("custom_nodes/ComfyUI-CLIPSeg"),
    Path("custom_nodes/ComfyUI_FaceAnalysis"),
    Path("models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
    Path("models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors"),
    Path("models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors"),
    Path("models/sams/sam_vit_b_01ec64.pth"),
]
for path in required:
    if not path.exists():
        raise SystemExit(f"missing required visual prompt stack artifact: {{path}}")
print("OK:visual_prompt_stack_files")
PY
"""
    with connect() as client:
        comfy_root = find_comfy_root(client)
        command = f"COMFY_ROOT={shlex.quote(comfy_root)} bash -lc {shlex.quote(script)}"
        code = run_remote_stream(client, command)
        if code != 0:
            raise SystemExit(code)


def start_temp_comfyui(_args) -> None:
    port = _args.port
    script = f"""
set -euo pipefail
cd "$COMFY_ROOT"
if python3 - <<'PY'
import socket
port = {port}
sock = socket.socket()
sock.settimeout(0.5)
raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
then
  echo "ComfyUI already listening on port {port}"
  exit 0
fi
log="$COMFY_ROOT/user/comfyui-port-{port}.log"
echo "Starting temporary ComfyUI backend on port {port}; log: $log"
nohup python3 main.py --listen 0.0.0.0 --port {port} --enable-manager > "$log" 2>&1 &
echo $! > "$COMFY_ROOT/user/comfyui-port-{port}.pid"
for i in $(seq 1 60); do
  if python3 - <<'PY'
import socket
port = {port}
sock = socket.socket()
sock.settimeout(0.5)
raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
  then
    echo "Temporary ComfyUI ready on port {port}"
    exit 0
  fi
  sleep 2
done
echo "Timed out waiting for temporary ComfyUI on port {port}" >&2
tail -n 120 "$log" >&2 || true
exit 1
"""
    with connect() as client:
        comfy_root = find_comfy_root(client)
        command = f"COMFY_ROOT={shlex.quote(comfy_root)} bash -lc {shlex.quote(script)}"
        _, out, err = run_remote(client, command, check=False)
        if out:
            print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")


def queue(_args) -> None:
    workflow_path = Path(_args.workflow)
    if not workflow_path.exists():
        raise SystemExit(f"Missing local workflow file: {workflow_path}")
    prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
    prompt_json = json.dumps(prompt)
    remote_script = r"""
import json
import sys
import time
import urllib.request

payload = json.load(sys.stdin)
wait = int(payload["wait"])
port = int(payload["port"])
token = ""
try:
    token = open("/app/ComfyUI/login/PASSWORD", encoding="utf-8").readline().strip()
except FileNotFoundError:
    pass

def request(path, data=None):
    url = f"http://127.0.0.1:{port}" + path
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)

queued = request("/prompt", {"prompt": payload["prompt"]})
prompt_id = queued["prompt_id"]
print("PROMPT_ID=" + prompt_id)

if wait <= 0:
    raise SystemExit(0)

deadline = time.time() + wait
while time.time() < deadline:
    history = request("/history/" + prompt_id)
    if prompt_id in history:
        item = history[prompt_id]
        status = item.get("status", {})
        print("STATUS=" + str(status.get("status_str", "unknown")))
        outputs = item.get("outputs", {})
        files = []
        for node in outputs.values():
            for image in node.get("images", []):
                files.append(image.get("subfolder", "") + "/" + image.get("filename", ""))
        for file in files:
            print("OUTPUT=" + file.lstrip("/"))
        raise SystemExit(0 if status.get("completed", False) else 1)
    time.sleep(10)

print("STATUS=timeout")
raise SystemExit(2)
"""
    payload = json.dumps({"prompt": prompt, "wait": _args.wait, "port": _args.port})
    with connect() as client:
        _, out, err = run_remote(
            client,
            f"python3 -c {shlex.quote(remote_script)}",
            check=False,
            stdin_data=payload,
        )
        if out:
            print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")


def download(_args) -> None:
    local_dir = Path(_args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with connect() as client:
                comfy_root = find_comfy_root(client)
                remote_path = _args.remote_path
                if not remote_path.startswith("/"):
                    remote_path = posixpath.join(comfy_root, "output", remote_path)
                local_path = local_dir / posixpath.basename(remote_path)
                sftp = client.open_sftp()
                try:
                    sftp.get(remote_path, str(local_path))
                finally:
                    sftp.close()
                print(local_path)
                return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(f"Download retry {attempt}/{attempts - 1} for {_args.remote_path}: {exc}", file=sys.stderr)
            time.sleep(2)
    raise SystemExit(f"Download failed after {attempts} attempts: {last_error}")


def run(_args) -> None:
    if not _args.command:
        raise SystemExit("Usage: simplepod.py run <remote command>")
    with connect() as client:
        command = " ".join(_args.command)
        _, out, err = run_remote(client, command, check=False)
        if out:
            print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")


def postprocess_skin_tone(_args) -> None:
    script = REMOTE_SKIN_TONE_POSTPROCESS.read_text(encoding="utf-8")
    with connect() as client:
        comfy_root = find_comfy_root(client)

        def resolve(remote_path: str) -> str:
            if remote_path.startswith("/"):
                return remote_path
            return posixpath.join(comfy_root, "output", remote_path)

        payload = json.dumps(
            {
                "image": resolve(_args.image),
                "candidate_mask": resolve(_args.candidate_mask),
                "face_mask": resolve(_args.face_mask),
                "output": resolve(_args.output),
                "refined_mask_output": resolve(_args.refined_mask_output),
                "threshold": _args.threshold,
                "min_region_pixels": _args.min_region_pixels,
                "dilate": _args.dilate,
                "blur": _args.blur,
                "strength": _args.strength,
            }
        )
        remote_script = (
            "import json, os, shlex, subprocess, sys, tempfile\n"
            "payload = json.load(sys.stdin)\n"
            f"script = {script!r}\n"
            "with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as handle:\n"
            "    handle.write(script)\n"
            "    path = handle.name\n"
            "try:\n"
            "    cmd = ['python3', path]\n"
            "    for key in ['image', 'candidate_mask', 'face_mask', 'output', 'refined_mask_output', 'threshold', 'min_region_pixels', 'dilate', 'blur', 'strength']:\n"
            "        cmd.extend([f'--{key.replace(\"_\", \"-\")}', str(payload[key])])\n"
            "    proc = subprocess.run(cmd, text=True, capture_output=True)\n"
            "    if proc.stdout:\n"
            "        print(proc.stdout, end='')\n"
            "    if proc.stderr:\n"
            "        print(proc.stderr, file=sys.stderr, end='')\n"
            "    raise SystemExit(proc.returncode)\n"
            "finally:\n"
            "    try:\n"
            "        os.unlink(path)\n"
            "    except FileNotFoundError:\n"
            "        pass\n"
        )
        _, out, err = run_remote(
            client,
            f"python3 -c {shlex.quote(remote_script)}",
            check=False,
            stdin_data=payload,
        )
        if out:
            print(out, end="")
        if err:
            print(err, file=sys.stderr, end="")


def install_node(_args) -> None:
    script = """
set -euo pipefail
if command -v node >/dev/null 2>&1; then
  echo "Node.js is already installed: $(node -v)"
  exit 0
fi
echo "Installing Node.js v22..."
curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh
bash /tmp/nodesource_setup.sh
apt-get install -y nodejs
echo "Node.js installed: $(node -v)"
"""
    with connect() as client:
        print("Checking/installing Node.js on SimplePod...")
        code = run_remote_stream(client, f"bash -lc {shlex.quote(script)}")
        if code != 0:
            raise SystemExit(code)


def deploy_app(_args) -> None:
    import tempfile
    import zipfile
    
    with connect() as client:
        comfy_root = find_comfy_root(client)
        app_dir = posixpath.join(posixpath.dirname(comfy_root), "faceswap_deploy")
        print(f"Deploying app to {app_dir} ...")
        run_remote(client, f"mkdir -p {shlex.quote(app_dir)}")
        
        sftp = client.open_sftp()
        try:
            for filename in [".env", "requirements.txt", "package.json", "package-lock.json", "vite.config.js", "spiderman.png", "superman.png", "superman adult.png", "subject_5 year curly.webp"]:
                local_path = ROOT / filename
                if local_path.exists():
                    sftp.put(str(local_path), posixpath.join(app_dir, filename))
            
            for directory in ["app", "scripts", "frontend", "workflows"]:
                local_dir = ROOT / directory
                if not local_dir.exists():
                    continue
                remote_dir = posixpath.join(app_dir, directory)
                run_remote(client, f"mkdir -p {shlex.quote(remote_dir)}")
                with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
                    with zipfile.ZipFile(tmp.name, "w") as zf:
                        for path in local_dir.rglob("*"):
                            if "__pycache__" in path.parts or path.name.startswith(".DS_Store") or "node_modules" in path.parts:
                                continue
                            if path.is_file():
                                zf.write(path, path.relative_to(local_dir))
                    remote_zip = posixpath.join(remote_dir, "upload.zip")
                    print(f"Uploading {directory} archive...")
                    sftp.put(tmp.name, remote_zip)
                    run_remote(client, f"cd {shlex.quote(remote_dir)} && python3 -m zipfile -e upload.zip . && rm upload.zip")
        finally:
            sftp.close()

        print("Building UI on SimplePod...")
        code = run_remote_stream(client, f"cd {shlex.quote(app_dir)} && npm install && npm run build")
        if code != 0:
            raise SystemExit("Failed to build UI remotely.")

        print("Installing Python requirements on SimplePod...")
        code, out, err = run_remote(client, f"cd {shlex.quote(app_dir)} && python3 -m pip install --break-system-packages -r requirements.txt", check=False)
        if code != 0:
            print(err, file=sys.stderr)
            raise SystemExit("Failed to install requirements.")
        print("App deployed successfully.")


def serve_app(_args) -> None:
    env = load_env()
    host = env.get("SIMPLEPOD_SSH_HOST", "<host>")
    port_env = env.get("SIMPLEPOD_SSH_PORT", "22")
    print(f"== SimplePod UI Serving ==")
    print(f"The UI is starting on the SimplePod at port 8000.")
    print(f"If the port is publicly exposed, access it at: http://{host}:8000")
    print(f"If not, run this command locally in a new terminal to tunnel:")
    print(f"  ssh -N -L 8000:127.0.0.1:8000 root@{host} -p {port_env}")
    print(f"Then open http://localhost:8000 in your browser.")
    print(f"===========================\n")
    
    with connect() as client:
        comfy_root = find_comfy_root(client)
        app_dir = posixpath.join(posixpath.dirname(comfy_root), "faceswap_deploy")
        command = f"cd {shlex.quote(app_dir)} && python3 scripts/run_deploy_app.py"
        run_remote_stream(client, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("profile").set_defaults(func=profile)
    sub.add_parser("deploy").set_defaults(func=deploy)
    sub.add_parser("deploy-instantid").set_defaults(func=deploy_instantid)
    sub.add_parser("deploy-instantid-crop").set_defaults(func=deploy_instantid_crop)
    sub.add_parser("deploy-swap-and-bake").set_defaults(func=deploy_swap_and_bake)
    sub.add_parser("deploy-visual-prompt-hybrid").set_defaults(func=deploy_visual_prompt_hybrid)
    sub.add_parser("init-auth").set_defaults(func=init_auth)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--port", type=int, default=8188)
    preflight_parser.set_defaults(func=preflight)
    instantid_preflight_parser = sub.add_parser("preflight-instantid")
    instantid_preflight_parser.add_argument("--port", type=int, default=8188)
    instantid_preflight_parser.add_argument("--crop-stitch", action="store_true")
    instantid_preflight_parser.set_defaults(func=preflight_instantid)
    visual_preflight_parser = sub.add_parser("preflight-visual-prompt")
    visual_preflight_parser.add_argument("--port", type=int, default=8188)
    visual_preflight_parser.set_defaults(func=preflight_visual_prompt)
    sub.add_parser("install-instantid").set_defaults(func=install_instantid)
    sub.add_parser("install-visual-prompt-stack").set_defaults(func=install_visual_prompt_stack)
    sub.add_parser("install-reactor").set_defaults(func=install_reactor)
    temp_parser = sub.add_parser("start-temp-comfyui")
    temp_parser.add_argument("--port", type=int, default=8190)
    temp_parser.set_defaults(func=start_temp_comfyui)
    queue_parser = sub.add_parser("queue")
    queue_parser.add_argument("--workflow", default=str(WORKFLOW))
    queue_parser.add_argument("--wait", type=int, default=300, help="seconds to wait for completion; use 0 to only submit")
    queue_parser.add_argument("--port", type=int, default=8188)
    queue_parser.set_defaults(func=queue)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("remote_path")
    download_parser.add_argument("--local-dir", default="test_outputs")
    download_parser.set_defaults(func=download)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=run)
    skin_parser = sub.add_parser("postprocess-skin-tone")
    skin_parser.add_argument("--image", required=True)
    skin_parser.add_argument("--candidate-mask", required=True)
    skin_parser.add_argument("--face-mask", required=True)
    skin_parser.add_argument("--output", required=True)
    skin_parser.add_argument("--refined-mask-output", required=True)
    skin_parser.add_argument("--threshold", type=int, default=32)
    skin_parser.add_argument("--min-region-pixels", type=int, default=2500)
    skin_parser.add_argument("--dilate", type=int, default=3)
    skin_parser.add_argument("--blur", type=float, default=5.0)
    skin_parser.add_argument("--strength", type=float, default=0.85)
    skin_parser.set_defaults(func=postprocess_skin_tone)
    sub.add_parser("install-node").set_defaults(func=install_node)
    sub.add_parser("deploy-app").set_defaults(func=deploy_app)
    sub.add_parser("serve-app").set_defaults(func=serve_app)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
