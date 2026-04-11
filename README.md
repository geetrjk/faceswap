# faceswap

Subject-on-character face swap workflow for ComfyUI with:
- automatic face detection,
- explicit intermediate previews,
- local-first run path (no SimplePod required).

## What is included

- `scripts/build_faceswap_workflow.py`: generates a ComfyUI API workflow JSON.
- `workflows/faceswap_subject_on_character_api.json`: ready-to-import prompt graph.
- `docs/runbook.md`: run steps for local ComfyUI and optional SimplePod usage.

## Workflow strategy

1. Load **subject** face image and **target** character image.
2. Run a low-denoise diffusion pass to harmonize lighting/style while preserving composition.
3. Perform face swap (subject identity onto target face region).
4. Run automatic face detection + face-detailer refinement.
5. Save both:
   - intermediate swapped result (`faceswap/final_*`)
   - refined result (`faceswap/refined_*`)

This keeps the first repo's core idea (diffusion + swap) while adding robust automatic detection/refinement and verification checkpoints.

## Generate or regenerate the workflow JSON

```bash
python scripts/build_faceswap_workflow.py
```

You can override assets/models:

```bash
python scripts/build_faceswap_workflow.py \
  --subject-image my_subject.jpg \
  --target-image my_target.png \
  --checkpoint flux1-dev-fp8.safetensors \
  --vae ae.safetensors
```

## Note about this environment

Direct GitHub outbound access was blocked in this container, so the workflow was implemented locally from the requested design constraints and ComfyUI node conventions rather than by downloading and editing the referenced JSON directly in-place.
