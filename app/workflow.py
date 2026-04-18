from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .config import IMAGE_SUFFIXES, Settings


def list_target_templates(settings: Settings) -> list[dict[str, str]]:
    templates: list[dict[str, str]] = []
    for path in sorted(settings.template_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lowered = path.name.lower()
        if lowered.startswith("subject") or "test_subject" in lowered:
            continue
        templates.append(
            {
                "name": path.name,
                "label": path.stem.replace("_", " ").replace("-", " ").title(),
            }
        )
    return templates


def resolve_template_path(settings: Settings, template_name: str) -> Path:
    path = (settings.template_dir / template_name).resolve()
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(template_name)
    return path


def load_workflow(settings: Settings) -> dict[str, Any]:
    return json.loads(settings.stable_workflow_api_path.read_text(encoding="utf-8"))


def build_job_workflow(
    settings: Settings,
    *,
    subject_filename: str,
    target_filename: str,
    job_id: str,
) -> dict[str, Any]:
    workflow = copy.deepcopy(load_workflow(settings))
    load_image_nodes = [node for node in workflow.values() if node.get("class_type") == "LoadImage"]
    if len(load_image_nodes) < 2:
        raise RuntimeError("Stable workflow does not expose the expected subject and target LoadImage nodes.")

    load_image_nodes[0]["inputs"]["image"] = subject_filename
    load_image_nodes[1]["inputs"]["image"] = target_filename

    for node in workflow.values():
        if node.get("class_type") != "SaveImage":
            continue
        original_prefix = node["inputs"]["filename_prefix"]
        stage_name = original_prefix.rstrip("/").split("/")[-1]
        node["inputs"]["filename_prefix"] = f"faceswap/deploy/{job_id}/{stage_name}"
    return workflow


def stage_input_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
