from __future__ import annotations

import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from .comfy import ComfyClient
from .config import get_settings
from .demo import DemoJobStore
from .database import Database, JobRecord
from .storage import R2Storage
from .workflow import build_job_workflow, list_target_templates, resolve_template_path, stage_input_file


settings = get_settings()
database = Database(settings)
storage = R2Storage(settings)
comfy = ComfyClient(settings)
demo_jobs = DemoJobStore(settings)

app = FastAPI(title="faceswap_deploy")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    same_site="lax",
    https_only=settings.secure_session_cookie,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginPayload(BaseModel):
    username: str
    password: str


@app.on_event("startup")
def startup() -> None:
    settings.local_upload_dir.mkdir(parents=True, exist_ok=True)
    database.ensure_schema()


def require_auth(request: Request) -> str:
    if not settings.auth_enabled():
        return ""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return str(user)


@app.get("/api/auth/session")
def auth_session(request: Request) -> dict[str, Any]:
    user = request.session.get("user")
    return {
        "enabled": settings.auth_enabled(),
        "authenticated": bool(user) or not settings.auth_enabled(),
        "username": user if user else None,
    }


@app.post("/api/auth/login")
def auth_login(payload: LoginPayload, request: Request) -> dict[str, Any]:
    if not settings.auth_enabled():
        request.session["user"] = "local"
        return {"authenticated": True, "username": "local"}
    valid_user = settings.app_username or ""
    valid_password = settings.app_password or ""
    if not (
        secrets.compare_digest(payload.username, valid_user)
        and secrets.compare_digest(payload.password, valid_password)
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    request.session["user"] = payload.username
    return {"authenticated": True, "username": payload.username}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict[str, Any]:
    request.session.clear()
    response.delete_cookie("session")
    return {"authenticated": False}


@app.get("/api/health")
def health(_user: str = Depends(require_auth)) -> dict[str, Any]:
    return {
        "shared_env_path": str(settings.shared_env_path),
        "shared_root": str(settings.shared_root),
        "stable_workflow_api_path": str(settings.stable_workflow_api_path),
        "stable_workflow_ui_path": str(settings.stable_workflow_ui_path),
        "mode": "demo" if settings.demo_mode else "live",
        "auth_enabled": settings.auth_enabled(),
        "database_ready": database.is_configured(),
        "storage_ready": storage.is_configured(),
        "missing": settings.missing_runtime_config(),
    }


@app.get("/api/templates")
def templates(_user: str = Depends(require_auth)) -> list[dict[str, str]]:
    return list_target_templates(settings)


@app.get("/api/templates/{template_name}/preview")
def template_preview(template_name: str, _user: str = Depends(require_auth)) -> FileResponse:
    try:
        path = resolve_template_path(settings, template_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Template not found.") from exc
    return FileResponse(path)


@app.get("/api/jobs")
def jobs(_user: str = Depends(require_auth)) -> list[dict[str, Any]]:
    if settings.demo_mode:
        return [_serialize_demo_job(job) for job in demo_jobs.list_jobs()]
    if not database.is_configured():
        return []
    return [_serialize_job(job) for job in database.list_jobs()]


@app.get("/api/jobs/{job_id}")
def job(job_id: str, _user: str = Depends(require_auth)) -> dict[str, Any]:
    if settings.demo_mode:
        record = demo_jobs.get_job(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _serialize_demo_job(record)
    if not database.is_configured():
        raise HTTPException(status_code=503, detail="Neon is not configured.")
    record = database.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _serialize_job(record)


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def artifact(job_id: str, artifact_name: str, _user: str = Depends(require_auth)) -> FileResponse:
    if settings.demo_mode:
        record = demo_jobs.get_job(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found.")
        for item in record["artifacts"]:
            if item.get("name") != artifact_name:
                continue
            local_path = item.get("local_path")
            if local_path and Path(local_path).exists():
                return FileResponse(local_path)
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if not database.is_configured():
        raise HTTPException(status_code=503, detail="Neon is not configured.")
    record = database.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    for item in record.artifacts:
        if item.get("name") != artifact_name:
            continue
        local_path = item.get("local_path")
        if local_path and Path(local_path).exists():
            return FileResponse(local_path)
    raise HTTPException(status_code=404, detail="Artifact not found.")


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    subject: UploadFile = File(...),
    target_template: str = Form(...),
    _user: str = Depends(require_auth),
) -> dict[str, Any]:
    try:
        target_path = resolve_template_path(settings, target_template)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Template not found.") from exc

    job_id = str(uuid.uuid4())
    safe_subject_name = _safe_filename(subject.filename or "subject-upload.png")
    upload_dir = settings.local_upload_dir / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    uploaded_subject_path = upload_dir / safe_subject_name

    with uploaded_subject_path.open("wb") as handle:
        shutil.copyfileobj(subject.file, handle)

    comfy_subject_name = f"deploy_{job_id}_subject{uploaded_subject_path.suffix.lower()}"
    comfy_target_name = f"deploy_{job_id}_target{target_path.suffix.lower()}"
    comfy_subject_path = settings.comfy_input_dir / comfy_subject_name
    comfy_target_path = settings.comfy_input_dir / comfy_target_name
    if settings.demo_mode:
        demo_jobs.create_job(
            job_id=job_id,
            subject_original_name=safe_subject_name,
            target_template=target_template,
            workflow_name=f"{settings.stable_workflow_api_path.name} (demo)",
        )
        background_tasks.add_task(_run_demo_job, job_id, uploaded_subject_path, target_path)
        record = demo_jobs.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=500, detail="Failed to create demo job.")
        return _serialize_demo_job(record)

    missing = settings.missing_runtime_config()
    if missing:
        raise HTTPException(status_code=503, detail={"missing": missing})

    stage_input_file(uploaded_subject_path, comfy_subject_path)
    stage_input_file(target_path, comfy_target_path)

    if storage.is_configured():
        subject_r2_key = f"jobs/{job_id}/uploads/{safe_subject_name}"
        subject_r2_url = storage.upload_file(uploaded_subject_path, subject_r2_key)
    else:
        subject_r2_key = None
        subject_r2_url = None


    database.create_job(
        job_id=job_id,
        subject_original_name=safe_subject_name,
        subject_r2_key=subject_r2_key,
        subject_r2_url=subject_r2_url,
        target_template=target_template,
        workflow_name=settings.stable_workflow_api_path.name,
    )

    background_tasks.add_task(_run_job, job_id, comfy_subject_name, comfy_target_name)
    record = database.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=500, detail="Failed to create job.")
    return _serialize_job(record)


def _run_job(job_id: str, comfy_subject_name: str, comfy_target_name: str) -> None:
    try:
        database.update_job(job_id, status="queueing")
        prompt = build_job_workflow(
            settings,
            subject_filename=comfy_subject_name,
            target_filename=comfy_target_name,
            job_id=job_id,
        )
        prompt_id = comfy.queue_prompt(prompt)
        database.update_job(job_id, status="running", comfy_prompt_id=prompt_id, error_message=None)
        history_item = comfy.wait_for_completion(prompt_id)
        artifacts = []
        for output in comfy.collect_output_files(history_item):
            local_path = Path(output["local_path"])
            if not local_path.exists():
                continue
            artifact_name = _artifact_name(output["subfolder"], output["filename"])
            r2_key = f"jobs/{job_id}/artifacts/{artifact_name}/{output['filename']}"
            if storage.is_configured():
                public_url = storage.upload_file(local_path, r2_key)
            else:
                public_url = None
            artifacts.append(
                {
                    "name": artifact_name,
                    "filename": output["filename"],
                    "subfolder": output["subfolder"],
                    "r2_key": r2_key if storage.is_configured() else None,
                    "url": public_url,
                    "local_path": str(local_path),
                }
            )
        database.update_job(job_id, status="completed", artifacts=artifacts, error_message=None)
    except Exception as exc:
        database.update_job(job_id, status="failed", error_message=str(exc))


def _run_demo_job(job_id: str, subject_path: Path, target_path: Path) -> None:
    try:
        demo_jobs.run_demo_job(job_id, subject_path, target_path)
    except Exception as exc:
        demo_jobs.update_job(job_id, status="failed", error_message=str(exc))


def _serialize_job(job: JobRecord) -> dict[str, Any]:
    artifacts = []
    for artifact in job.artifacts:
        artifact_name = artifact["name"]
        proxy_path = f"/api/jobs/{job.id}/artifacts/{artifact_name}"
        artifacts.append(
            {
                **artifact,
                "proxy_url": proxy_path,
                "display_url": artifact.get("url") or proxy_path,
            }
        )
    return {
        "id": job.id,
        "status": job.status,
        "subject_original_name": job.subject_original_name,
        "subject_r2_key": job.subject_r2_key,
        "subject_r2_url": job.subject_r2_url,
        "target_template": job.target_template,
        "workflow_name": job.workflow_name,
        "comfy_prompt_id": job.comfy_prompt_id,
        "artifacts": artifacts,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _serialize_demo_job(job: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for artifact in job.get("artifacts", []):
        artifact_name = artifact["name"]
        proxy_path = f"/api/jobs/{job['id']}/artifacts/{artifact_name}"
        artifacts.append(
            {
                **artifact,
                "proxy_url": proxy_path,
                "display_url": artifact.get("url") or proxy_path,
            }
        )
    return {**job, "artifacts": artifacts}


def _artifact_name(subfolder: str, filename: str) -> str:
    last_part = (subfolder or filename).rstrip("/").split("/")[-1] or "output"
    return _safe_filename(last_part)


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {".", "-", "_"} else "_" for char in value)
    return cleaned.strip("._") or "file"


static_dir = settings.frontend_dist_dir if settings.frontend_dist_dir.exists() else settings.app_root / "app" / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
