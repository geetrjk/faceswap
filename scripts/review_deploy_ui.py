#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parent / "faceswap"
STABLE_API = SHARED_ROOT / "workflows" / "stable" / "visual_prompt_hybrid_v1_api.json"
STABLE_UI = SHARED_ROOT / "workflows" / "stable" / "visual_prompt_hybrid_v1_ui.json"
SUBJECT_IMAGE = ROOT / "subject_5 year curly.webp"
TARGET_TEMPLATE = "superman.png"
DEFAULT_SCREENSHOT_DIR = ROOT / "tmp_ui_review"


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def check_python_dependencies() -> None:
    python_path = ROOT / ".venv" / "bin" / "python"
    if not python_path.exists():
        raise SystemExit("Missing .venv/bin/python. Create the virtualenv first.")
    result = run([str(python_path), "-m", "pip", "list"])
    print(result.stdout)


def check_node_dependencies() -> None:
    package_json = ROOT / "package.json"
    node_modules = ROOT / "node_modules"
    if not package_json.exists() or not node_modules.exists():
        raise SystemExit("Missing package.json or node_modules. Run npm install first.")
    result = run(["npm", "list", "--depth=0"])
    print(result.stdout)


def validate_workflows() -> None:
    run(["python3", "-m", "json.tool", str(STABLE_API)])
    run(["python3", "-m", "json.tool", str(STABLE_UI)])
    print("Validated stable workflow JSON files.")


def build_frontend() -> None:
    result = run(["npm", "run", "build"])
    print(result.stdout)


def compile_backend() -> None:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".pycache")
    result = run(["python3", "-m", "compileall", "app", "scripts/run_deploy_app.py"], env=env)
    print(result.stdout)


def demo_api_run() -> None:
    env = os.environ.copy()
    env["DEPLOY_UI_DEMO_MODE"] = "1"
    env["DEPLOY_APP_USERNAME"] = "reviewer"
    env["DEPLOY_APP_PASSWORD"] = "review-pass"
    script = """
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
subject = Path("subject_5 year curly.webp")
session = client.get("/api/auth/session").json()
client.post("/api/auth/login", json={"username": "reviewer", "password": "review-pass"})
health = client.get("/api/health").json()
with subject.open("rb") as handle:
    created = client.post(
        "/api/jobs",
        files={"subject": (subject.name, handle, "image/webp")},
        data={"target_template": "superman.png"},
    ).json()
job = client.get(f"/api/jobs/{created['id']}").json()
print(json.dumps({
    "auth_enabled": session["enabled"],
    "health_mode": health["mode"],
    "job_status": job["status"],
    "artifacts": [artifact["name"] for artifact in job["artifacts"]],
}, indent=2))
"""
    result = run([str(ROOT / ".venv" / "bin" / "python"), "-c", script], env=env)
    print(result.stdout)


def live_queue_run(wait_seconds: int) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "scripts/simplepod.py",
        "queue",
        "--workflow",
        str(STABLE_API),
        "--wait",
        str(wait_seconds),
    ]
    result = run(command)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def capture_ui_screenshots(output_dir: Path, port: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "-d", str(ROOT / "frontend" / "dist")],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(port)
        node_script = f"""
const path = require('path');
const {{ chromium }} = require('playwright');
(async() => {{
  const browser = await chromium.launch({{ headless: true }});
  const appUrl = 'http://127.0.0.1:{port}/index.html?review=1';
  const desktop = await browser.newPage({{ viewport: {{ width: 1510, height: 1180 }}, deviceScaleFactor: 1 }});
  await desktop.goto(appUrl, {{ waitUntil: 'networkidle' }});
  await desktop.getByLabel('Username').fill('reviewer');
  await desktop.getByLabel('Password').fill('review-pass');
  await desktop.getByRole('button', {{ name: 'Sign In' }}).click();
  await desktop.waitForTimeout(300);
  await desktop.screenshot({{ path: {json.dumps(str(output_dir / "01-overview.png"))}, fullPage: true }});
  await desktop.locator('input[type="file"]').setInputFiles(path.resolve({json.dumps(str(SUBJECT_IMAGE))}));
  await desktop.waitForTimeout(500);
  await desktop.screenshot({{ path: {json.dumps(str(output_dir / "02-with-upload.png"))}, fullPage: true }});
  await desktop.getByRole('button', {{ name: 'Queue Full Body Swap' }}).click();
  await desktop.waitForTimeout(2200);
  await desktop.screenshot({{ path: {json.dumps(str(output_dir / "03-completed.png"))}, fullPage: true }});
  const mobile = await browser.newPage({{ viewport: {{ width: 430, height: 1320 }}, isMobile: true }});
  await mobile.goto(appUrl, {{ waitUntil: 'networkidle' }});
  await mobile.getByLabel('Username').fill('reviewer');
  await mobile.getByLabel('Password').fill('review-pass');
  await mobile.getByRole('button', {{ name: 'Sign In' }}).click();
  await mobile.waitForTimeout(300);
  await mobile.locator('input[type="file"]').setInputFiles(path.resolve({json.dumps(str(SUBJECT_IMAGE))}));
  await mobile.getByRole('button', {{ name: 'Queue Full Body Swap' }}).click();
  await mobile.waitForTimeout(2200);
  await mobile.screenshot({{ path: {json.dumps(str(output_dir / "04-mobile.png"))}, fullPage: true }});
  await browser.close();
}})();
"""
        result = run(["node", "-e", node_script])
        if result.stdout:
            print(result.stdout)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    print(f"Saved screenshots under {output_dir}")


def wait_for_port(port: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for localhost:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe review runner for the faceswap deploy UI.")
    parser.add_argument("--skip-build", action="store_true", help="Skip npm run build.")
    parser.add_argument("--skip-screenshots", action="store_true", help="Skip Playwright UI screenshots.")
    parser.add_argument("--live-queue", action="store_true", help="Queue the shared stable workflow on the live ComfyUI backend.")
    parser.add_argument("--live-wait", type=int, default=900, help="Seconds to wait when --live-queue is used.")
    parser.add_argument("--port", type=int, default=4173, help="Local review server port for UI screenshots.")
    parser.add_argument("--screenshot-dir", default=str(DEFAULT_SCREENSHOT_DIR), help="Directory for review screenshots.")
    args = parser.parse_args()

    check_python_dependencies()
    check_node_dependencies()
    compile_backend()
    validate_workflows()
    if not args.skip_build:
        build_frontend()
    demo_api_run()
    if not args.skip_screenshots:
        capture_ui_screenshots(Path(args.screenshot_dir), args.port)
    if args.live_queue:
        live_queue_run(args.live_wait)


if __name__ == "__main__":
    main()
