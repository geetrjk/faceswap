# Visual Prompt Hybrid Saved Result

Saved after prompt `ed427fcd-b0ff-4c4f-871c-28a883f08b9b` completed successfully on the main `8188` backend.

Files:

- `visual_prompt_hybrid_experiment_api.json`
- `visual_prompt_hybrid_experiment_ui.json`
- `target_head_mask_00001_.png`
- `generated_head_00001_.png`
- `reactor_snap_00001_.png`
- `bake_mask_00001_.png`
- `final_00001_.png`

Validation note: the live backend did not expose PuLID, IP-Adapter, or Impact/SAM nodes, so this runnable sidecar uses the same high-level automation shape with available nodes: FaceSegmentation target mask, SDXL head fill, ReActor likeness snap, and low-denoise full-image bake. It avoided the gray and green abstract artifacts, but it is not a full PuLID/IP-Adapter/SAM validation.
