# Project Agents

Codex entrypoint for the faceswap ComfyUI pipeline.

## Goal

Build a clean, reusable SimplePod-ready ComfyUI base pipeline for subject-to-character face swap experiments.

## Operating Rules

- Keep API and UI workflows paired:
  - `workflows/faceswap_subject_on_character_api.json` is the `/prompt` automation artifact.
  - `workflows/faceswap_subject_on_character_ui.json` is the browser-runnable ComfyUI graph.
- Regenerate both workflow files with `python scripts/build_faceswap_workflow.py` after changing builder defaults or graph settings.
- Deploy both workflow files with `scripts/simplepod.py deploy`.
- Validate both formats before handoff: `python3 -m json.tool workflows/faceswap_subject_on_character_api.json` and `python3 -m json.tool workflows/faceswap_subject_on_character_ui.json`.
- Keep the UI workflow output nodes in sync with the API workflow checkpoints so human-in-the-loop debugging can inspect the same stages.
- Profile a new pod before provisioning or queueing: GPU, disk, Python, pip, and ComfyUI root path.
- Prefer `scripts/simplepod.py` for remote checks and file sync. Avoid ad hoc SSH commands unless the helper is missing the needed action.
- Keep remote execution simple: deploy workflow/assets, verify required nodes/files, then queue from ComfyUI or via a small API script once the graph is validated.
- Record repeated mistakes and fixes in `docs/known_mistakes.md` before ending a session.

## Minimal Agent Setup

- `.agents/skills/remote-operator/SKILL.md`: SimplePod profiling and deployment checklist.
- `docs/runbook.md`: human-facing run steps.
- `docs/STATUS.md`: current handoff state.
- `docs/known_mistakes.md`: short reusable mistake ledger.
