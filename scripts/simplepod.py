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
WORKFLOW = ROOT / "workflows" / "faceswap_subject_on_character_api.json"
UI_WORKFLOW = ROOT / "workflows" / "faceswap_subject_on_character_ui.json"
INPUTS = [
    ROOT / "subject_5 year curly.webp",
    ROOT / "superman.png",
]
REQUIRED_NODES = [
    "ReActorFaceSwap",
]
REQUIRED_MODEL_PATHS = [
    "models/insightface/inswapper_128.onnx",
    "models/facerestore_models/GFPGANv1.4.pth",
]


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
        run_remote(client, f"python -c {shlex.quote(script)}", stdin_data=f"{password}\n{username}\n")


def preflight(_args) -> None:
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
            "python - <<'PY'\n"
            "import json, urllib.request\n"
            "token_path = '/app/ComfyUI/login/PASSWORD'\n"
            "token = ''\n"
            "try:\n"
            "    token = open(token_path, encoding='utf-8').readline().strip()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "req = urllib.request.Request('http://127.0.0.1:8188/object_info')\n"
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
token = ""
try:
    token = open("/app/ComfyUI/login/PASSWORD", encoding="utf-8").readline().strip()
except FileNotFoundError:
    pass

def request(path, data=None):
    url = "http://127.0.0.1:8188" + path
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
    payload = json.dumps({"prompt": prompt, "wait": _args.wait})
    with connect() as client:
        _, out, err = run_remote(
            client,
            f"python -c {shlex.quote(remote_script)}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("profile").set_defaults(func=profile)
    sub.add_parser("deploy").set_defaults(func=deploy)
    sub.add_parser("init-auth").set_defaults(func=init_auth)
    sub.add_parser("preflight").set_defaults(func=preflight)
    queue_parser = sub.add_parser("queue")
    queue_parser.add_argument("--workflow", default=str(WORKFLOW))
    queue_parser.add_argument("--wait", type=int, default=300, help="seconds to wait for completion; use 0 to only submit")
    queue_parser.set_defaults(func=queue)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("remote_path")
    download_parser.add_argument("--local-dir", default="test_outputs")
    download_parser.set_defaults(func=download)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
