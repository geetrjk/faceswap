from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import Settings


class ComfyClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        response = self._request("/prompt", {"prompt": prompt})
        return response["prompt_id"]

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 5,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            history = self._request(f"/history/{prompt_id}")
            item = history.get(prompt_id)
            if item:
                status = item.get("status", {})
                if status.get("completed"):
                    return item
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI prompt failed: {json.dumps(status)}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}.")

    def collect_output_files(self, history_item: dict[str, Any]) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        outputs = history_item.get("outputs", {})
        for node_id, node_output in outputs.items():
            for image in node_output.get("images", []):
                subfolder = image.get("subfolder", "")
                filename = image.get("filename", "")
                local_path = (self._settings.comfy_output_dir / subfolder / filename).resolve()
                files.append(
                    {
                        "node_id": str(node_id),
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": image.get("type", "output"),
                        "local_path": str(local_path),
                    }
                )
        return files

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(f"{self._settings.comfy_api_url}{path}", data=body)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        token = self._token()
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI API request failed for {path}: {exc.code} {details}") from exc

    def _token(self) -> str:
        token_path: Path = self._settings.comfy_token_file
        if not token_path.exists():
            return ""
        return token_path.read_text(encoding="utf-8").splitlines()[0].strip()
