from __future__ import annotations

import mimetypes
from pathlib import Path

import boto3

from .config import Settings


class R2Storage:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def is_configured(self) -> bool:
        return bool(
            self._settings.r2_account_id
            and self._settings.r2_access_key_id
            and self._settings.r2_secret_access_key
            and self._settings.r2_bucket
        )

    def upload_file(self, path: Path, key: str, content_type: str | None = None) -> str | None:
        if not self.is_configured():
            raise RuntimeError("R2 storage is not configured.")
        client = self._get_client()
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        client.upload_file(
            str(path),
            self._settings.r2_bucket,
            key,
            ExtraArgs={"ContentType": guessed_type},
        )
        return self.public_url(key)

    def public_url(self, key: str) -> str | None:
        if self._settings.r2_public_base_url:
            return f"{self._settings.r2_public_base_url}/{key}"
        return None

    def _get_client(self):
        if self._client is None:
            endpoint = f"https://{self._settings.r2_account_id}.r2.cloudflarestorage.com"
            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self._settings.r2_access_key_id,
                aws_secret_access_key=self._settings.r2_secret_access_key,
                region_name="auto",
            )
        return self._client
