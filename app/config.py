from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ROOT = ROOT.parent / "faceswap"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    app_root: Path
    shared_root: Path
    shared_env_path: Path
    stable_workflow_api_path: Path
    stable_workflow_ui_path: Path
    template_dir: Path
    local_state_dir: Path
    local_upload_dir: Path
    comfy_api_url: str
    comfy_input_dir: Path
    comfy_output_dir: Path
    comfy_token_file: Path
    neon_database_url: str | None
    r2_account_id: str | None
    r2_access_key_id: str | None
    r2_secret_access_key: str | None
    r2_bucket: str | None
    r2_public_base_url: str | None
    demo_mode: bool
    frontend_dist_dir: Path
    app_username: str | None
    app_password: str | None
    session_secret: str
    secure_session_cookie: bool

    def missing_runtime_config(self) -> list[str]:
        missing: list[str] = []
        if not self.stable_workflow_api_path.exists():
            missing.append(f"workflow:{self.stable_workflow_api_path}")
        if not self.stable_workflow_ui_path.exists():
            missing.append(f"workflow:{self.stable_workflow_ui_path}")
        if not self.template_dir.exists():
            missing.append(f"templates:{self.template_dir}")
        if not self.neon_database_url:
            missing.append("NEON_DATABASE_URL")
        if not self.r2_account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not self.r2_bucket:
            missing.append("R2_BUCKET")
        return missing

    def auth_enabled(self) -> bool:
        return bool(self.app_username and self.app_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    shared_root = Path(os.environ.get("FACE_SWAP_SHARED_ROOT", DEFAULT_SHARED_ROOT)).resolve()
    shared_env_path = Path(
        os.environ.get("FACE_SWAP_SHARED_ENV", str(shared_root / ".env"))
    ).resolve()
    env_values = {**_read_env_file(shared_env_path), **os.environ}

    stable_api = Path(
        env_values.get(
            "STABLE_WORKFLOW_API_PATH",
            str(shared_root / "workflows" / "stable" / "visual_prompt_hybrid_v1_api.json"),
        )
    ).resolve()
    stable_ui = Path(
        env_values.get(
            "STABLE_WORKFLOW_UI_PATH",
            str(shared_root / "workflows" / "stable" / "visual_prompt_hybrid_v1_ui.json"),
        )
    ).resolve()
    local_state_dir = Path(
        env_values.get("DEPLOY_LOCAL_STATE_DIR", str(ROOT / "var"))
    ).resolve()
    local_upload_dir = local_state_dir / "uploads"

    return Settings(
        app_root=ROOT,
        shared_root=shared_root,
        shared_env_path=shared_env_path,
        stable_workflow_api_path=stable_api,
        stable_workflow_ui_path=stable_ui,
        template_dir=Path(env_values.get("TARGET_TEMPLATE_DIR", str(shared_root))).resolve(),
        local_state_dir=local_state_dir,
        local_upload_dir=local_upload_dir,
        comfy_api_url=env_values.get("COMFYUI_API_URL", "http://127.0.0.1:8188").rstrip("/"),
        comfy_input_dir=Path(env_values.get("COMFYUI_INPUT_DIR", "/app/ComfyUI/input")).resolve(),
        comfy_output_dir=Path(env_values.get("COMFYUI_OUTPUT_DIR", "/app/ComfyUI/output")).resolve(),
        comfy_token_file=Path(
            env_values.get("COMFYUI_TOKEN_FILE", "/app/ComfyUI/login/PASSWORD")
        ).resolve(),
        neon_database_url=env_values.get("NEON_DATABASE_URL") or env_values.get("DATABASE_URL"),
        r2_account_id=env_values.get("R2_ACCOUNT_ID"),
        r2_access_key_id=env_values.get("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=env_values.get("R2_SECRET_ACCESS_KEY"),
        r2_bucket=env_values.get("R2_BUCKET"),
        r2_public_base_url=env_values.get("R2_PUBLIC_BASE_URL", "").rstrip("/") or None,
        demo_mode=env_values.get("DEPLOY_UI_DEMO_MODE", "").lower() in {"1", "true", "yes", "on"},
        frontend_dist_dir=(ROOT / "frontend" / "dist").resolve(),
        app_username=env_values.get("DEPLOY_APP_USERNAME"),
        app_password=env_values.get("DEPLOY_APP_PASSWORD"),
        session_secret=env_values.get("DEPLOY_APP_SESSION_SECRET", "dev-only-session-secret"),
        secure_session_cookie=env_values.get("DEPLOY_APP_SECURE_COOKIE", "").lower() in {"1", "true", "yes", "on"},
    )
