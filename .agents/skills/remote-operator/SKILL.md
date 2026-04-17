---
name: Remote Operator
description: SimplePod ComfyUI profiling, deployment, and preflight checks for this faceswap repo.
---

# Remote Operator Skill

Use this skill when working against a SimplePod or other remote ComfyUI instance.

## First Checks

Before installing or queueing anything on a new pod, run:

```bash
.venv/bin/python scripts/simplepod.py profile
```

This checks the GPU, disk, Python, pip, and likely ComfyUI root paths.

## Deploy

Regenerate the workflow locally, then deploy workflow and input assets:

```bash
.venv/bin/python scripts/build_faceswap_workflow.py
.venv/bin/python scripts/simplepod.py deploy
.venv/bin/python scripts/simplepod.py init-auth
```

## Preflight

Before queueing, verify the remote graph dependencies:

```bash
.venv/bin/python scripts/simplepod.py preflight
```

Required custom nodes:

- `ReActorFaceSwap`

Required model files:

- `models/insightface/inswapper_128.onnx`
- `models/facerestore_models/GFPGANv1.4.pth`

## Human In The Loop

If node registration is stale after installing custom nodes, do not keep polling. Ask for a real SimplePod restart, then run `.venv/bin/python scripts/simplepod.py preflight` again.
