# Status

Updated: 2026-04-12

## Current State

- API workflow exists at `workflows/faceswap_subject_on_character_api.json`.
- UI workflow exists at `workflows/faceswap_subject_on_character_ui.json`.
- Workflow builder exists at `scripts/build_faceswap_workflow.py` and is the source of truth for both workflow JSON files.
- Current local inputs are `subject_5 year curly.webp` and `superman.png`.
- SimplePod env keys are expected in `.env`; secrets stay untracked.

## Next Run

1. Install local helper dependency: `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.
2. Run `.venv/bin/python scripts/simplepod.py profile`.
3. Run `.venv/bin/python scripts/build_faceswap_workflow.py`.
4. Run `.venv/bin/python scripts/simplepod.py deploy`.
5. Run `.venv/bin/python scripts/simplepod.py init-auth`.
6. Restart the pod from SimplePod if custom nodes were installed but `/object_info` does not list them.
7. Run `.venv/bin/python scripts/simplepod.py preflight`.

## Current Risk

The workflow now depends only on ReActor plus `inswapper_128.onnx` and `GFPGANv1.4.pth`. The first remote run should be treated as a base swap validation run, not a final-quality face swap attempt.

## Latest Remote Check

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
