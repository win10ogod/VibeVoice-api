# Repository Guidelines

## Project Structure & Module Organization
- `vibevoice/` core library
  - `modular/` model, schedulers, streamers
  - `processor/` tokenizer/processor utilities
  - `finetune/` LoRA + training
- `vibevoice_api/` OpenAI‑compatible audio API (FastAPI)
- `web/` browser UIs (SSE/Binary console)
- `scripts/` CLI tools (HTTP tests, stress, JS demos)
- `demo/` runnable examples + `demo/voices/` samples
- `voice_map.yaml` default voice alias mapping (see `config/voice_map.yaml.sample`)
- `tests/` add new tests mirroring package paths

## Build, Test, and Development Commands
- Install (editable): `uv pip install -e .` or `pip install -e .`
- Run API server: `python -m vibevoice_api.server --model_path vibevoice/VibeVoice-1.5B --port 8000`
- Web console: open `http://127.0.0.1:8000/web/console.html`
- Pure‑HTTP test: `python scripts/api_audio_speech_test.py --base_url http://127.0.0.1:8000 --response_format mp3`
- Stress (pip openai): `python scripts/stress_vibevoice_api.py --concurrency 4 --requests 16 --include_compressed`
- Tests: `pytest -q`

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, type hints where reasonable.
- Names: modules `snake_case`, classes `PascalCase`, functions/vars `snake_case`.
- Prefer `logging` over prints; avoid global state; keep public APIs stable.
- Place model code in `vibevoice/modular/`, processor/data in `vibevoice/processor/`, API in `vibevoice_api/`.

## Testing Guidelines
- Framework: `pytest`; short, hermetic tests (no network/model downloads in CI).
- Layout: `tests/<pkg>/test_*.py` (e.g., `tests/vibevoice_api/test_server.py`).
- Add unit tests for processors and light shape/contract checks for models.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(api): add SSE audio`) with concise, imperative messages.
- PRs: include description, motivation, usage notes, risks; link issues; attach short audio/text samples when relevant.
- CI checklist: run a smoke test of the API or scripts; `pytest -q` must pass; update docs when flags/UX change.

## Security & Configuration Tips
- Do not commit secrets or large assets. Use env vars (e.g., `VIBEVOICE_FFMPEG`, `VIBEVOICE_MODEL`).
- Voice aliases via `voice_map.yaml` or `VIBEVOICE_VOICE_MAP`.
- Streaming/encoding knobs via env (examples):
  - `VIBEVOICE_SSE_CHUNK_BYTES` (SSE aggregation), `VIBEVOICE_OPUS_CONTAINER` (webm/ogg),
  - `VIBEVOICE_OPUS_VBR`/`OPUS_APPLICATION`/`OPUS_FRAME_DURATION`, `VIBEVOICE_AAC_PROFILE`/`AAC_MODE`/`AAC_Q`.

