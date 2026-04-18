from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .config import Settings


class DemoJobStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(
        self,
        *,
        job_id: str,
        subject_original_name: str,
        target_template: str,
        workflow_name: str,
    ) -> None:
        now = _now()
        self._jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "subject_original_name": subject_original_name,
            "subject_r2_key": None,
            "subject_r2_url": None,
            "target_template": target_template,
            "workflow_name": workflow_name,
            "comfy_prompt_id": f"demo-{job_id[:8]}",
            "artifacts": [],
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }

    def update_job(self, job_id: str, **fields: Any) -> None:
        record = self._jobs[job_id]
        record.update(fields)
        record["updated_at"] = _now()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 12) -> list[dict[str, Any]]:
        jobs = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
        return jobs[:limit]

    def run_demo_job(self, job_id: str, subject_path: Path, target_path: Path) -> None:
        self.update_job(job_id, status="queueing")
        time.sleep(0.15)
        self.update_job(job_id, status="running")
        time.sleep(0.3)

        demo_dir = self._settings.local_state_dir / "demo_artifacts" / job_id
        demo_dir.mkdir(parents=True, exist_ok=True)

        saved_dir = self._settings.shared_root / "saved_results" / "visual_prompt_hybrid_20260413"
        final_source = saved_dir / "final_00001_.png"
        generated_source = saved_dir / "generated_head_00001_.png"
        mask_source = saved_dir / "target_head_mask_00001_.png"

        subject_copy = demo_dir / f"subject_{subject_path.name}"
        target_copy = demo_dir / f"target_{target_path.name}"
        final_copy = demo_dir / "final_demo_result.png"
        generated_copy = demo_dir / "generated_head.png"
        mask_copy = demo_dir / "target_head_mask.png"

        shutil.copy2(subject_path, subject_copy)
        shutil.copy2(target_path, target_copy)
        shutil.copy2(final_source, final_copy)
        shutil.copy2(generated_source, generated_copy)
        shutil.copy2(mask_source, mask_copy)

        artifacts = [
            _artifact("subject_input", subject_copy),
            _artifact("target_template", target_copy),
            _artifact("generated_head", generated_copy),
            _artifact("target_head_mask", mask_copy),
            _artifact("final_result", final_copy),
        ]
        self.update_job(job_id, status="completed", artifacts=artifacts, error_message=None)


def _artifact(name: str, path: Path) -> dict[str, str]:
    return {
        "name": name,
        "filename": path.name,
        "subfolder": path.parent.name,
        "r2_key": "",
        "url": "",
        "local_path": str(path),
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
