# Status

Updated: 2026-04-13

## Current State

- API workflow exists at `workflows/faceswap_subject_on_character_api.json`.
- UI workflow exists at `workflows/faceswap_subject_on_character_ui.json`.
- Workflow builder exists at `scripts/build_faceswap_workflow.py` and is the source of truth for both workflow JSON files.
- Current local inputs are `subject_5 year curly.webp` and `superman.png`.
- SimplePod env keys are expected in `.env`; secrets stay untracked.

## Next Run

1. For a fresh SimplePod server, run `scripts/setup_simplepod_instantid.sh`.
2. Restart the ComfyUI/SimplePod backend when the script pauses for restart.
3. Let the script finish the ReActor, InstantID, and crop-stitch preflight checks.
4. For manual UI work, load `faceswap_subject_on_character_ui`, `instantid_subject_pose_style_ui`, or `instantid_crop_stitch_experiment_ui` from the `faceswap` workflow folder.

## Current Risk

The ReActor baseline is validated, but the InstantID/crop-stitch branch is still experimental. Setup is now reproducible; quality remains blocked on stronger structure guidance for the stylized target, because sparse face keypoints plus SDXL inpaint still produce abstract colored patches in the face region.

## Latest Remote Check

- Main backend `8188` now passes crop-stitch preflight after the real ComfyUI/SimplePod restart:
  - Required InstantID/crop-stitch nodes are live, including `ApplyInstantIDAdvanced`, `FaceSegmentation`, `CropMask`, `GrowMask`, and `ImageCrop`.
  - Added stronger crop-stitch structural guidance with built-in `Canny` plus `ControlNetApplyAdvanced`.
  - Installed `models/controlnet/controlnet-canny-sdxl-1.0-small.safetensors` via the resumable `install-instantid` path.
  - Crop-stitch preflight on `8188` now checks the Canny node, ControlNet apply node, and SDXL Canny ControlNet model.
- Regenerated and deployed `workflows/instantid_crop_stitch_experiment_api.json` and `workflows/instantid_crop_stitch_experiment_ui.json` with a new saved `target_crop_canny` inspection output.
- Canny-guided crop-stitch prompt `558aec99-ea7e-48fb-ab04-99578fe0ef4d` completed successfully on `8188` and produced `faceswap/instantid_crop_stitch/final_00004_.png`.
  - Local inspection set: `test_outputs/instantid_crop_stitch_20260413_canny/`.
  - The Canny guide captured much stronger target crop structure than the five-point keypoint image, but the raw inpaint still had the same green/colorful mask-like face artifact.
- Diagnostic prompt `fd874876-9d99-46e2-88d4-382bc72cbf2b` with InstantID pose/control strength lowered to `0.05` and Canny strength raised to `0.65` completed successfully.
  - Local raw crop: `test_outputs/instantid_crop_stitch_20260413_canny_pose005/crop_inpaint_raw_00001_.png`.
  - Local final: `test_outputs/instantid_crop_stitch_20260413_canny_pose005/final_canny_pose005_00001_.png`.
  - Result was not better; the colored artifact became harsher even though the edge guide changed facial structure.
- Updated quality diagnosis: Canny gives a better structure signal and proves the new structural branch is executable, but it does not fix the colored abstract inpaint artifact by itself. Next branch should try a non-face-edge structural/control path or a different identity/inpaint stack, not more prompt-only tuning.
- Added a separate swap-and-bake sidecar experiment without changing the current InstantID pipeline:
  - Builder: `scripts/build_swap_and_bake_workflow.py`
  - API: `workflows/swap_and_bake_experiment_api.json`
  - UI: `workflows/swap_and_bake_experiment_ui.json`
  - Deploy helper: `.venv/bin/python scripts/simplepod.py deploy-swap-and-bake`
- Swap-and-bake uses ReActor as a first pass, saves the swapped image and face mask for inspection, then runs a low-denoise full-image SDXL bake from the swapped image. The template's masked `VAEEncodeForInpaint` form was tested first but produced a gray face plate at denoise `0.20` and `0.45`, so the checked-in workflow uses plain `VAEEncode` for the bake step while keeping the mask checkpoint visible.
- Swap-and-bake prompt `41cfb464-9a93-414e-826d-40e9d3021000` completed successfully on `8188`.
  - Local inspection set: `test_outputs/swap_and_bake_20260413_full_bake/`.
  - Result avoided the gray/abstract InstantID artifact and produced a coherent baked image, though it still inherits ReActor's identity limitations.
- Saved the current swap-and-bake workflow/results under `saved_results/swap_and_bake_20260413/`.
- Added a separate visual-prompt hybrid sidecar experiment without changing the current InstantID or swap-and-bake pipelines:
  - Builder: `scripts/build_visual_prompt_hybrid_workflow.py`
  - API: `workflows/visual_prompt_hybrid_experiment_api.json`
  - UI: `workflows/visual_prompt_hybrid_experiment_ui.json`
  - Deploy helper: `.venv/bin/python scripts/simplepod.py deploy-visual-prompt-hybrid`
- The requested reference architecture uses PuLID, IP-Adapter Plus, and SAM/Impact nodes, but the live `8188` backend did not expose those nodes. The checked-in workflow is therefore a runnable environment fallback with the same high-level automation shape: FaceSegmentation target head mask, SDXL head fill, ReActor likeness snap, and low-denoise full-image bake.
- Visual-prompt hybrid fallback prompt `ed427fcd-b0ff-4c4f-871c-28a883f08b9b` completed successfully on `8188`.
  - Local inspection set: `test_outputs/visual_prompt_hybrid_20260413/`.
  - Saved workflow/results: `saved_results/visual_prompt_hybrid_20260413/`.
  - Result avoided the gray face plate and green/abstract artifact, but it is not yet a full PuLID/IP-Adapter/SAM validation.
- Fresh-pod setup path was hardened after the restarted container lost nodes/models:
  - `scripts/simplepod.py install-*` now streams remote output, times out stalled git operations, falls back to GitHub zip archives, checks minimum model sizes, resumes partial downloads with `curl -C -`, and normalizes nested AntelopeV2 archive extraction.
  - `scripts/setup_simplepod_instantid.sh` is the preferred repeatable setup entry point for ReActor plus InstantID/crop-stitch.
  - Repeat install checks now finish quickly when files are present: ReActor model checks and InstantID SDXL/adapter/controlnet/model archive checks skip completed files.
- Current restarted pod state:
  - `8188` main backend sees all model files but has a stale node registry until a real SimplePod/ComfyUI restart.
  - Temporary backend `8190` passes ReActor, InstantID, and crop-stitch preflight.
- Installed and selected `sd_xl_base_1.0_inpainting_0.1.safetensors` as the default InstantID checkpoint for both InstantID builders/workflows.
- Crop-stitch prompt `09bd8e07-725e-40e7-a568-552c5583679a` on `8190` with SDXL base still produced a colorful abstract raw inpaint patch; local inspection set: `test_outputs/instantid_crop_stitch_20260413_setupfix/`.
- Crop-stitch prompt `093c650c-c34c-4278-8f5d-86ef8efd3510` on `8190` with the SDXL inpaint checkpoint still produced a colorful abstract raw inpaint patch; local outputs: `test_outputs/instantid_crop_stitch_20260413_inpaint_ckpt/`.
- Diagnostic prompt `5592c07f-988e-4bc0-ae3b-559b769628b2` with pose/control strength lowered to `0.05` still produced the same abstract colored patch; local outputs: `test_outputs/instantid_crop_stitch_20260413_pose005/`.
- Current quality diagnosis: crop/mask/stitch wiring is good and containment works, but the target cartoon face keypoints are only a sparse five-point guide and the InstantID/inpaint branch is not producing a coherent face. The next quality branch should add a stronger structural guide such as DWPose/OpenPose/depth/lineart rather than more prompt-only or weight-only nudging.

- Restarted pod was fresh: RTX 3060 12 GB, ComfyUI at `/app/ComfyUI`, but InstantID custom nodes and model files were missing.
- Reinstalled `ComfyUI_InstantID`, `ComfyUI_IPAdapter_plus` (not currently used by the regenerated graph), and `ComfyUI_FaceAnalysis`; offline imports now expose `ApplyInstantIDAdvanced` and `FaceSegmentation`.
- Installed SDXL base, InstantID adapter/controlnet, AntelopeV2, and Buffalo-L face-analysis models.
- Regenerated and deployed the InstantID workflow as a masked inpaint/composite graph: target face mask -> masked InstantID -> target latent inpaint -> final composite over the unchanged target.
- Current InstantID state: the main ComfyUI backend on port `8188` still has a stale node registry, but the temporary backend on port `8190` has all InstantID nodes and required model files registered.
- API prompt `e20d990e-473c-4887-8e29-80696f620cea` completed on port `8190` and produced all InstantID checkpoints under `faceswap/instantid/iter_20260412/`.
- Low-denoise diagnostic prompt `1112c97c-4cef-432b-aa11-beb6bd077662` completed but preserved a flat gray face plate.
- High-denoise diagnostic prompt `467858a4-a434-42a5-a056-caacdff683fb` completed and generated a face-like region, but eyes/mouth were misaligned and target keypoint guidance remained weak.
- No-keypoints prompt `8dad8371-8a1a-4134-b5b9-4454e100cb39`, subject-keypoints prompt `903216d9-ab96-43b2-8289-b3e000615627`, and global-InstantID/no-mask prompt `12603e3b-4f81-4f27-834f-5b01f460b820` all completed but produced worse geometry/leakage.
- Attempted to fetch an SDXL inpaint checkpoint, but the remote download repeatedly stalled; the incomplete remote file was renamed to `/app/ComfyUI/models/checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors.partial` so it is not selected accidentally.
- Regenerated and deployed the paired InstantID API/UI workflows with diagnostic defaults: denoise `0.86`, InstantID weight `1.35`, pose/control strength `0.2`, face mask grow `16`, and stronger negative prompt terms for gray/blank face artifacts.
- Remote UI workflow verification: 26 nodes, 38 links, 6 `SaveImage` nodes, with visible preview nodes for target keypoints, raw face inpaint, and final composite.
- Current deployed API smoke prompt `dc7fd7f5-64cc-4815-a561-7ab360b96bf9` completed successfully on port `8190` and produced `faceswap/instantid/final_00002_.png`; local copy is `test_outputs/instantid_deployed_20260412/final_00002_.png`.
- Current handoff direction: stop blind API iteration and use the deployed UI workflow for human inspection of the target face mask, target keypoints, raw inpaint, and final composite.
- Added a separate crop-stitch InstantID experiment without replacing the current InstantID or ReActor workflows:
  - Builder: `scripts/build_instantid_crop_stitch_workflow.py`
  - API: `workflows/instantid_crop_stitch_experiment_api.json`
  - UI: `workflows/instantid_crop_stitch_experiment_ui.json`
  - Deploy helper: `.venv/bin/python scripts/simplepod.py deploy-instantid-crop`
- Crop-stitch preflight passed on temporary backend port `8190` with `--crop-stitch`; the main browser backend still needs a real ComfyUI/SimplePod restart before InstantID UI workflows run on port `8188`.
- Crop-stitch prompt `8eea93fc-3d0e-4091-8369-75eac2e4813a` proved dynamic crop/stitch wiring works but used too-large an edit mask and produced major color artifacts.
- Revised crop-stitch prompt `76ade8ea-370e-412c-8de4-eaf8ed54a0da` separated the generous crop-region mask from a smaller edit mask and completed successfully, but still produced abstract/color artifacts in the masked face region with SDXL base.
- Denoise `0.65` prompt `42396162-ad8d-49c4-9c8c-f92013ec71b7` reduced the affected area slightly but did not solve quality; current diagnosis is that crop/stitch improves containment but the SDXL base inpaint core remains the limiting issue.

- Local helper dependency installed into `.venv`.
- Pod profile: RTX 3060 with 12 GB VRAM, 100 GB disk, ComfyUI at `/app/ComfyUI`.
- ReActor cloned to `/app/ComfyUI/custom_nodes/ComfyUI-ReActor`.
- ReActor dependencies, `inswapper_128.onnx`, and `GFPGANv1.4.pth` installed on the pod.
- Authenticator password file initialized for API token use after restart.
- Slim ReActor-only workflow deployed to `/app/ComfyUI/user/default/workflows/faceswap/faceswap_subject_on_character_api.json`.
- Offline import check confirmed `ReActorFaceSwap` is available in the installed node package.
- After pod restart, preflight confirmed `ReActorFaceSwap` in live `/object_info`.
- Base workflow prompt `6a6477af-75bc-4e65-8bd9-55fe01fa3b83` completed with status `success`.
- Remote output: `/app/ComfyUI/output/faceswap/final_00001_.png`.
- Local downloaded output: `test_outputs/final_00001_.png`.
- Visual review: base ReActor swap succeeded against the Superman target; output is a pipeline milestone, not final quality.
- Variant prompts completed successfully:
  - no restore: `872a6f73-ed08-4292-a1f7-2b878bfb1497`, output `faceswap/variant_no_restore_00001_.png`.
  - GFPGAN 0.7: `193e1254-f569-4900-bab3-f684566be906`, output `faceswap/variant_restore_07_00001_.png`.
  - GFPGAN 1.0: `4141b45e-797b-4c3e-b203-1621fe3a2731`, output `faceswap/variant_restore_10_00001_.png`.
  - GFPGAN 1.0 + FaceBoost: `07cd0ce8-e816-4ec8-887f-fa877abc094f`, output `faceswap/variant_face_boost_00001_.png`.
- Promoted default workflow to GFPGAN 1.0 + FaceBoost and redeployed it.
- Canonical boosted prompt `fc5eee3d-aa50-4d7f-8ea0-582a3e7fdf6d` completed with status `success`.
- Canonical boosted output: `/app/ComfyUI/output/faceswap/final_00002_.png`.
- Local boosted output: `test_outputs/final_00002_.png`.
- Added human-in-the-loop output nodes for subject input, target input, plain no-restore swap, and boosted final.
- Verified all output branches by changing prefixes in prompt `3c607cc4-c973-44df-97fd-af2af80f2cab`; outputs written under `/app/ComfyUI/output/faceswap/hitl_intermediate_verify/` and `/app/ComfyUI/output/faceswap/final_hitl_all_verify_00001_.png`.
- Deployed updated default workflow with HITL output checkpoints to `/app/ComfyUI/user/default/workflows/faceswap/faceswap_subject_on_character_api.json`.
- Browser issue diagnosed: `faceswap_subject_on_character_api` is an API prompt, not a UI graph, so ComfyUI showed no output nodes when it was opened in the browser.
- Added and deployed UI-native workflow `/app/ComfyUI/user/default/workflows/faceswap/faceswap_subject_on_character_ui.json`.
- Remote UI workflow verification: 13 graph nodes, 13 links, and 8 output nodes.
- API smoke test prompt `f32db979-dc5b-43b4-942d-d4a942c54612` completed and produced all expected checkpoints, including `/app/ComfyUI/output/faceswap/final_00003_.png`.
