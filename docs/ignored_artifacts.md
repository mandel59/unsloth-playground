# Gitignored artifacts needed for scripts

This repository ignores generated artifacts and local dependencies (see
`.gitignore`). The items below are required to run `scripts/` or to follow the
training/merge workflow.

## Python environment (`.venv`)

- Preferred (uses `pyproject.toml` + `uv.lock`): `uv sync`
- Alternative: `python -m venv .venv` then `pip install -e .`

## IDS source database (`moji.db`)

- Required by: `scripts/ids_experiment.py prepare`
- Fetch with Deno:

```
deno run --allow-write=moji.db --unstable-raw-imports scripts/get_mojidata.ts
```

## Fonts (`fonts/Jigmo*.ttf`)

- Required by: `scripts/ids_experiment.py prepare` (default `--fonts`)
- Place these files under `fonts/`:
  - `fonts/Jigmo.ttf`
  - `fonts/Jigmo2.ttf`
  - `fonts/Jigmo3.ttf`
- The Jigmo font bundle can be downloaded from:
  `https://kamichikoichi.github.io/jigmo/Jigmo-20250912.zip`
- Archives can be stored anywhere; this repo uses `downloads/` by convention
  (it is gitignored).
- If you use different fonts, pass `--fonts` to the `prepare` command.

## `tools/llama.cpp` (GGUF tooling)

- Required by: `scripts/merge_gguf_qwen3vl.py` and
  `tools/llama.cpp/convert_hf_to_gguf.py`
- Clone the repo into `tools/llama.cpp`:

```
git clone https://github.com/ggerganov/llama.cpp tools/llama.cpp
```

## Generated outputs (ignored by design)

- `outputs/ids_dataset` is created by `python scripts/ids_experiment.py prepare`
- `outputs/IDS-LoRA-Qwen3-VL-2B*` is created by
  `python scripts/ids_experiment.py train`
- `outputs/merged-*.gguf` and `outputs/*.Modelfile` are created by the merge and
  Ollama steps
- `mlflow.db` is created by MLflow during training/evaluation
- `unsloth_compiled_cache` is created automatically by Unsloth
