from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import Settings


@dataclass
class JobRecord:
    id: str
    status: str
    subject_original_name: str
    subject_r2_key: str | None
    subject_r2_url: str | None
    target_template: str
    workflow_name: str
    comfy_prompt_id: str | None
    artifacts: list[dict[str, Any]]
    error_message: str | None
    created_at: str
    updated_at: str


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_configured(self) -> bool:
        return bool(self._settings.neon_database_url)

    def _connect(self) -> psycopg.Connection[Any]:
        if not self._settings.neon_database_url:
            raise RuntimeError("NEON_DATABASE_URL is not configured.")
        return psycopg.connect(self._settings.neon_database_url, row_factory=dict_row)

    def ensure_schema(self) -> None:
        if not self.is_configured():
            return
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                create table if not exists deploy_jobs (
                    id uuid primary key,
                    status text not null,
                    subject_original_name text not null,
                    subject_r2_key text,
                    subject_r2_url text,
                    target_template text not null,
                    workflow_name text not null,
                    comfy_prompt_id text,
                    artifacts_json text not null default '[]',
                    error_message text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                )
                """
            )
            conn.commit()

    def create_job(
        self,
        *,
        job_id: str,
        subject_original_name: str,
        subject_r2_key: str | None,
        subject_r2_url: str | None,
        target_template: str,
        workflow_name: str,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into deploy_jobs (
                    id,
                    status,
                    subject_original_name,
                    subject_r2_key,
                    subject_r2_url,
                    target_template,
                    workflow_name
                )
                values (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id,
                    "queued",
                    subject_original_name,
                    subject_r2_key,
                    subject_r2_url,
                    target_template,
                    workflow_name,
                ),
            )
            conn.commit()

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        comfy_prompt_id: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        error_message: str | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = %s"]
        values: list[Any] = [datetime.now(timezone.utc)]
        if status is not None:
            assignments.append("status = %s")
            values.append(status)
        if comfy_prompt_id is not None:
            assignments.append("comfy_prompt_id = %s")
            values.append(comfy_prompt_id)
        if artifacts is not None:
            assignments.append("artifacts_json = %s")
            values.append(json.dumps(artifacts))
        if error_message is not None or status == "failed":
            assignments.append("error_message = %s")
            values.append(error_message)
        values.append(job_id)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"update deploy_jobs set {', '.join(assignments)} where id = %s",
                values,
            )
            conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("select * from deploy_jobs where id = %s", (job_id,))
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_jobs(self, limit: int = 12) -> list[JobRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select *
                from deploy_jobs
                order by created_at desc
                limit %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: dict[str, Any]) -> JobRecord:
        return JobRecord(
            id=str(row["id"]),
            status=row["status"],
            subject_original_name=row["subject_original_name"],
            subject_r2_key=row.get("subject_r2_key"),
            subject_r2_url=row.get("subject_r2_url"),
            target_template=row["target_template"],
            workflow_name=row["workflow_name"],
            comfy_prompt_id=row.get("comfy_prompt_id"),
            artifacts=json.loads(row.get("artifacts_json") or "[]"),
            error_message=row.get("error_message"),
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
        )


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
