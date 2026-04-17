# InstantID Subject-First Experiment

This is a separate workflow path from the ReActor baseline.

## Goal

- Subject image: identity anchor.
- Target/cartoon image: pose, head direction, broad composition, and weak style reference.
- Output: preserve the subject's face, age cues, head shape, and hairstyle more strongly than the target face.

## Files

- Builder: `scripts/build_instantid_workflow.py`
- API workflow: `workflows/instantid_subject_pose_style_api.json`
- UI workflow: `workflows/instantid_subject_pose_style_ui.json`
- Crop-stitch builder: `scripts/build_instantid_crop_stitch_workflow.py`
- Crop-stitch API workflow: `workflows/instantid_crop_stitch_experiment_api.json`
- Crop-stitch UI workflow: `workflows/instantid_crop_stitch_experiment_ui.json`

The ReActor baseline remains unchanged:

- `scripts/build_faceswap_workflow.py`
- `workflows/faceswap_subject_on_character_api.json`
- `workflows/faceswap_subject_on_character_ui.json`

## Graph Strategy

1. Load the subject and target images.
2. Load an SDXL checkpoint.
3. Use InstantID to condition identity from the subject image.
4. Extract target face keypoints and feed them to `ApplyInstantIDAdvanced` as `image_kps`.
5. Generate a soft target-face mask with FaceAnalysis `FaceSegmentation`.
6. Feed that mask into `ApplyInstantIDAdvanced` and `VAEEncodeForInpaint`.
7. Sample from the target image latent, not an empty latent, with denoise below full strength.
8. Composite the decoded inpaint back over the original target using the same soft mask.
9. Save the result under `faceswap/instantid/final_*`.

Current diagnostic defaults favor getting a real generated face region instead of preserving the gray inpaint fill. They are not final quality settings:

- InstantID identity weight: `1.35`
- Target pose/control strength: `0.2`
- Face mask area: `face+forehead (if available)`
- Face mask grow: `16`
- Face mask blur: `31`
- Inpaint denoise: `0.86`

If the edit bleeds into the body/template:

- lower `--denoise`
- lower `--face-mask-grow`
- raise `--face-mask-blur` if the boundary is too hard

If the target dominates age, skull shape, or hairstyle inside the mask:

- lower `--pose-strength`
- raise `--instantid-weight` slightly
- add more subject-preservation language to `--positive-prompt`
- add target-face leakage terms to `--negative-prompt`

## Required Custom Nodes

Install and restart ComfyUI before queueing:

- `ComfyUI_InstantID`
- `ComfyUI_FaceAnalysis`

The workflow also uses built-in ComfyUI nodes such as `CheckpointLoaderSimple`, `ControlNetLoader`, `CLIPTextEncode`, `VAEEncodeForInpaint`, `KSampler`, `VAEDecode`, `MaskToImage`, `ImageCompositeMasked`, `PreviewImage`, and `SaveImage`.

## Required Models

Expected default local model paths:

- `models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors`
- `models/checkpoints/sd_xl_base_1.0.safetensors` is still installed as a fallback baseline checkpoint.
- `models/instantid/ip-adapter.bin`
- `models/controlnet/instantid_controlnet.safetensors`
- `models/insightface/models/antelopev2/1k3d68.onnx`
- `models/insightface/models/antelopev2/2d106det.onnx`
- `models/insightface/models/antelopev2/genderage.onnx`
- `models/insightface/models/antelopev2/glintr100.onnx`
- `models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx`
- `models/insightface/models/buffalo_l/1k3d68.onnx`
- `models/insightface/models/buffalo_l/2d106det.onnx`
- `models/insightface/models/buffalo_l/det_10g.onnx`
- `models/insightface/models/buffalo_l/genderage.onnx`
- `models/insightface/models/buffalo_l/w600k_r50.onnx`

If your installed custom nodes expect different filenames, regenerate the workflow with the matching arguments:

```bash
python3 scripts/build_instantid_workflow.py \
  --checkpoint your_sdxl_checkpoint.safetensors \
  --instantid-model your_instantid_adapter.bin \
  --instantid-controlnet your_instantid_controlnet.safetensors
```

## SimplePod Commands

Generate, deploy, and preflight the experimental workflow:

```bash
python3 scripts/build_instantid_workflow.py
.venv/bin/python scripts/simplepod.py deploy-instantid
.venv/bin/python scripts/simplepod.py preflight-instantid
```

Queue the experimental API workflow after preflight passes:

```bash
.venv/bin/python scripts/simplepod.py queue \
  --workflow workflows/instantid_subject_pose_style_api.json \
  --wait 600
```

## VRAM Notes

- 12 GB: possible for a masked inpaint pass, but expect to reduce steps, close other workloads, or lower denoise if memory is tight.
- 16 GB: recommended minimum for smoother SDXL + InstantID testing.
- 24 GB: preferred for full 1024-class SDXL resolution, higher step counts, and later local refinement/crop-stitch experiments.

## First 12 GB Smoke Test

The current `896x640` settings completed successfully on the RTX 3060 12 GB pod through a temporary ComfyUI backend on port `8190`.

Outputs:

- `faceswap/instantid/intermediate/target_face_keypoints_00001_.png`
- `faceswap/instantid/final_00001_.png`

The first result preserved the subject's curly hair and childlike face structure much better than the ReActor baseline. Follow-up masked inpaint runs showed two failure modes:

- Low denoise (`0.42` to `0.58`) can leave a gray/flat inpaint plate over the target face.
- Higher denoise (`0.86`) generates a face-like region, but the target keypoint image is too sparse and the eyes/mouth land poorly.

The next useful inspection path is the UI workflow, not more blind API prompt nudging. Load `instantid_subject_pose_style_ui` in ComfyUI and inspect the saved target face mask, target face keypoints, raw face inpaint, and final composite nodes.

## Crop-Stitch Experiment

The crop-stitch variant evaluates a more controlled alternative:

1. Generate a generous target head/face region mask.
2. Crop the target image to that region so the sampler sees local context rather than the whole body/background.
3. Shrink the crop-local mask before InstantID/inpaint so the crop stays generous but the actual edit remains localized.
4. Run InstantID on the crop.
5. Composite the edited crop back into the original full target using the crop-local edit mask.

This is better for containment: the target body, cape, and background are protected outside the crop paste region, and the UI exposes `target_crop`, `target_crop_region_mask`, `target_crop_edit_mask`, `target_crop_keypoints`, `crop_inpaint_raw`, `crop_composite`, and `final`.

It is not yet a full quality fix. The smoke runs completed successfully, but SDXL base and the SDXL inpaint checkpoint produced colorful/abstract artifacts in the masked face region.

The current crop-stitch branch adds stronger structural guidance from the target crop:

- `Canny` extracts an edge image from the crop.
- `target_crop_canny` is saved as a UI/API inspection checkpoint.
- `ControlNetApplyAdvanced` applies `controlnet-canny-sdxl-1.0-small.safetensors` after InstantID conditioning and before the crop sampler.

This makes the structural signal much richer than the sparse cartoon face keypoints, and the graph runs successfully on the main `8188` backend. It still does not eliminate the colored mask-like raw inpaint artifact, so the next branch should try a non-face-edge structural/control path or a different identity/inpaint stack rather than more prompt-only tuning.

## Reused From The Baseline

- Same subject and target image defaults.
- Same API/UI workflow pairing pattern.
- Same intermediate output habit for human-in-the-loop debugging.
- Same SimplePod helper style, but under separate `deploy-instantid` and `preflight-instantid` commands.

ReActor is intentionally not part of the main identity path here. If used later, it should be a separate late-stage diagnostic/correction branch only.
