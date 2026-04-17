#!/usr/bin/env python3
"""Run the visual prompt hybrid workflow against multiple subject images."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_visual_prompt_hybrid_workflow.py"
SIMPLEPOD_SCRIPT = ROOT / "scripts" / "simplepod.py"
TEST_SUBJECTS_DIR = ROOT / "test_subjects"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(name).stem.lower()).strip("_")


def run(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def parse_outputs(stdout: str) -> list[str]:
    return [line.split("=", 1)[1].strip() for line in stdout.splitlines() if line.startswith("OUTPUT=")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", default=str(TEST_SUBJECTS_DIR))
    parser.add_argument("--target-image", default="superman.png")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--wait", type=int, default=600)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    subjects_dir = Path(args.subjects_dir)
    subjects = sorted(
        path
        for path in subjects_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not subjects:
        raise SystemExit(f"No subject images found in {subjects_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_slug = slugify(args.target_image)
    local_dir = Path(args.local_dir or ROOT / "test_outputs" / f"visual_prompt_subject_matrix_{target_slug}_{stamp}")
    local_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_setup:
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "profile"])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "install-reactor"])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "install-visual-prompt-stack"])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "preflight-visual-prompt", "--port", str(args.port)])
        run([sys.executable, str(BUILD_SCRIPT)])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "deploy-visual-prompt-hybrid"])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "init-auth"])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "start-temp-comfyui", "--port", str(args.port)])
        run([sys.executable, str(SIMPLEPOD_SCRIPT), "preflight-visual-prompt", "--port", str(args.port)])

    results: list[dict[str, str | list[str]]] = []
    for subject in subjects:
        slug = slugify(subject.name)
        run_dir = f"faceswap/visual_prompt_hybrid/subject_matrix_{target_slug}_{stamp}/{slug}"
        workflow_path = local_dir / f"{slug}_api.json"

        run(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--output",
                str(workflow_path),
                "--ui-output",
                str(local_dir / f"{slug}_ui.json"),
                "--subject-image",
                subject.name,
                "--target-image",
                args.target_image,
                "--filename-prefix",
                f"{run_dir}/final",
                "--intermediate-prefix",
                f"{run_dir}/intermediate",
            ]
        )

        queued = run(
            [
                sys.executable,
                str(SIMPLEPOD_SCRIPT),
                "queue",
                "--workflow",
                str(workflow_path),
                "--wait",
                str(args.wait),
                "--port",
                str(args.port),
            ],
            capture_output=True,
        )
        print(queued.stdout, end="")
        if queued.stderr:
            print(queued.stderr, file=sys.stderr, end="")

        outputs = parse_outputs(queued.stdout)
        pre_skin = next(path for path in outputs if path.endswith("/intermediate/pre_skin_harmonize_00001_.png"))
        candidate_mask = next(path for path in outputs if path.endswith("/intermediate/target_skin_mask_00001_.png"))
        face_mask = next(path for path in outputs if path.endswith("/intermediate/inner_face_mask_00001_.png"))
        harmonized_remote = f"{run_dir}/final_postprocess_00001_.png"
        refined_mask_remote = f"{run_dir}/intermediate/target_skin_mask_refined_00001_.png"
        post = run(
            [
                sys.executable,
                str(SIMPLEPOD_SCRIPT),
                "postprocess-skin-tone",
                "--image",
                pre_skin,
                "--candidate-mask",
                candidate_mask,
                "--face-mask",
                face_mask,
                "--output",
                harmonized_remote,
                "--refined-mask-output",
                refined_mask_remote,
            ],
            capture_output=True,
        )
        print(post.stdout, end="")
        if post.stderr:
            print(post.stderr, file=sys.stderr, end="")
        outputs.extend([harmonized_remote, refined_mask_remote])

        subject_dir = local_dir / slug
        subject_dir.mkdir(parents=True, exist_ok=True)
        for remote_path in outputs:
            run(
                [
                    sys.executable,
                    str(SIMPLEPOD_SCRIPT),
                    "download",
                    remote_path,
                    "--local-dir",
                    str(subject_dir),
                ]
            )
        results.append(
            {
                "subject": subject.name,
                "target_image": args.target_image,
                "outputs": outputs,
                "local_dir": str(subject_dir),
            }
        )

    (local_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {local_dir}")


if __name__ == "__main__":
    main()
