#!/usr/bin/env bash
# First-run SimplePod setup for the ReActor baseline plus InstantID experiments.
#
# Safe to rerun: scripts/simplepod.py install-* commands check existing custom
# nodes and model sizes, resume partial downloads, and stream remote progress.
#
# Requires .env with SIMPLEPOD_SSH_HOST, SIMPLEPOD_SSH_PORT, SIMPLEPOD_SSH_USER,
# SIMPLEPOD_PASSWORD, and SIMPLEPOD_COMFYUI_URL.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_PY=".venv/bin/python"

echo "== Local helper environment =="
if [[ ! -x "$VENV_PY" ]]; then
  "$PYTHON" -m venv .venv
fi
"$VENV_PY" -m pip install -r requirements.txt

echo "== Profile SimplePod =="
"$VENV_PY" scripts/simplepod.py profile

echo "== Generate workflows =="
"$VENV_PY" scripts/build_faceswap_workflow.py
"$VENV_PY" scripts/build_instantid_workflow.py
"$VENV_PY" scripts/build_instantid_crop_stitch_workflow.py

echo "== Install remote nodes and models =="
"$VENV_PY" scripts/simplepod.py install-reactor
"$VENV_PY" scripts/simplepod.py install-instantid
"$VENV_PY" scripts/simplepod.py init-auth

echo "== Deploy workflows and inputs =="
"$VENV_PY" scripts/simplepod.py deploy
"$VENV_PY" scripts/simplepod.py deploy-instantid
"$VENV_PY" scripts/simplepod.py deploy-instantid-crop

cat <<'MSG'

========================================================================
Restart required:
  Restart the ComfyUI/SimplePod backend now, then press Enter here.

Why:
  ComfyUI only loads custom node classes at startup. Without this restart,
  the browser UI can still show missing InstantID class_type errors even
  when the files were installed correctly.
========================================================================
MSG
read -r

echo "== Preflight live backend =="
"$VENV_PY" scripts/simplepod.py preflight
"$VENV_PY" scripts/simplepod.py preflight-instantid
"$VENV_PY" scripts/simplepod.py preflight-instantid --crop-stitch

cat <<'MSG'

Setup complete.

Manual UI workflows:
  - faceswap_subject_on_character_ui
  - instantid_subject_pose_style_ui
  - instantid_crop_stitch_experiment_ui
MSG
