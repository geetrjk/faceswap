# Swap And Bake Saved Result

Saved after prompt `41cfb464-9a93-414e-826d-40e9d3021000` completed successfully on the main `8188` backend.

Files:

- `swap_and_bake_experiment_api.json`
- `swap_and_bake_experiment_ui.json`
- `reactor_swap_00002_.png`
- `bake_mask_00002_.png`
- `final_00002_.png`

Validation note: the checked-in swap-and-bake workflow uses a ReActor first pass, saves the face mask for inspection, then uses a low-denoise full-image SDXL bake. This avoided the gray face plate seen with the masked `VAEEncodeForInpaint` bake.
