# faceswap

Subject-on-character face swap workflow for ComfyUI with:
- ReActor face swap,
- explicit intermediate previews,
- local-first run path (no SimplePod required).

## What is included

- `scripts/build_faceswap_workflow.py`: generates a ComfyUI API workflow JSON.
- `scripts/build_instantid_workflow.py`: generates a separate experimental SDXL + InstantID workflow.
- `scripts/simplepod.py`: profiles, deploys, and preflights the SimplePod ComfyUI instance.
- `workflows/faceswap_subject_on_character_api.json`: API prompt for automated queueing.
- `workflows/faceswap_subject_on_character_ui.json`: browser-runnable ComfyUI graph for manual review.
- `workflows/instantid_subject_pose_style_api.json`: experimental API prompt for subject-first InstantID generation.
- `workflows/instantid_subject_pose_style_ui.json`: browser-runnable experimental InstantID graph.
- `.agents/skills/remote-operator/SKILL.md`: reusable remote-operation checklist.
- `docs/runbook.md`: run steps for local ComfyUI and SimplePod usage.
- `docs/instantid_experiment.md`: setup notes for the subject-first InstantID alternative.
- `docs/known_mistakes.md`: compact ledger of mistakes and fixes to avoid repeating.

## Workflow strategy

1. Load **subject** face image and **target** character image.
2. Perform a ReActor swap from subject identity onto the target face region.
3. Apply one lightweight face-restore pass with `GFPGANv1.4.pth`.
4. Save the base swap output as `faceswap/final_*`.

This is intentionally smaller than the earlier diffusion/detailer idea. On a 12 GB SimplePod GPU, the first milestone is a reliable base swap; diffusion cleanup can be added after that works.

## Experimental InstantID path

The ReActor workflow remains the baseline. For subject-first generation that tries to keep the subject's age cues, head shape, and hairstyle while taking pose/style from the target, use the separate InstantID builder:

```bash
python3 scripts/build_instantid_workflow.py
.venv/bin/python scripts/simplepod.py deploy-instantid
.venv/bin/python scripts/simplepod.py preflight-instantid
```

This path requires additional SDXL base/inpaint, InstantID, ControlNet, AntelopeV2, and Buffalo-L face-analysis models. See `docs/instantid_experiment.md`.

## Generate or regenerate the workflow JSON

```bash
python scripts/build_faceswap_workflow.py
```

This writes both the API prompt and the UI graph. Keep them together in reviews and commits.

You can override assets/models:

```bash
python scripts/build_faceswap_workflow.py \
  --subject-image my_subject.jpg \
  --target-image my_target.png \
  --swap-model inswapper_128.onnx \
  --face-restore-model GFPGANv1.4.pth
```

## SimplePod quick path

The `.env` file should define:

```bash
SIMPLEPOD_SSH_HOST=...
SIMPLEPOD_SSH_PORT=22
SIMPLEPOD_SSH_USER=root
SIMPLEPOD_PASSWORD=...
SIMPLEPOD_COMFYUI_URL=...
```

Install the helper dependency in a local venv, profile the pod, deploy files, then preflight:

For a fresh SimplePod server, use the setup script:

```bash
scripts/setup_simplepod_instantid.sh
```

It installs local helper dependencies, installs ReActor and InstantID requirements on the pod, deploys all workflow variants, pauses for the required ComfyUI/SimplePod restart, then runs preflight checks. Remote model downloads are size-checked and resumable, so rerunning the script should skip completed files and resume partial files instead of starting over.

Manual equivalent for the ReActor baseline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/simplepod.py profile
.venv/bin/python scripts/build_faceswap_workflow.py
.venv/bin/python scripts/simplepod.py deploy
.venv/bin/python scripts/simplepod.py init-auth
.venv/bin/python scripts/simplepod.py preflight
.venv/bin/python scripts/simplepod.py queue --wait 300
```

The default workflow uses `GFPGANv1.4.pth` at full visibility plus ReActor FaceBoost. Use `--no-face-boost` when regenerating if you want the simpler base swap only.

For human-in-the-loop review in ComfyUI, the workflow now writes:

- `faceswap/intermediate/subject_input_*`
- `faceswap/intermediate/target_input_*`
- `faceswap/intermediate/plain_swap_*`
- `faceswap/final_*`

Load `faceswap_subject_on_character_ui` from the ComfyUI workflow browser when running manually. The `_api` file is for `scripts/simplepod.py queue` and will appear empty if opened as a UI graph.

## Minimum SimplePod spec

Recommended minimum for this base ReActor pipeline:

- GPU: NVIDIA with 12 GB VRAM minimum; 16 GB+ preferred for later diffusion cleanup.
- RAM: 24 GB minimum; 32 GB preferred.
- Disk: 40 GB minimum; 80 GB+ preferred once ReActor, ONNX Runtime, swap/restore models, and cache files are included.
- Image: Python 3.10+ with CUDA/PyTorch support and ComfyUI already installed or installable.

For an easier first run, choose a 16 GB VRAM GPU and treat 12 GB as the budget/debug floor.

## Note about this environment

Direct GitHub outbound access was blocked in this container, so the workflow was implemented locally from the requested design constraints and ComfyUI node conventions rather than by downloading and editing the referenced JSON directly in-place.
