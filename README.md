# Unsloth IDS LoRA (Qwen3-VL-2B)

This project fine-tunes a Qwen3-VL-2B-Instruct model with LoRA to convert
Hanzi images into Ideographic Description Sequences (IDS). It includes data
preparation, training, evaluation with MLflow logging, and optional export to
Ollama via a merged GGUF model.

## What this repo contains

- `scripts/ids_experiment.py`: end-to-end pipeline for dataset prep, training,
  and evaluation (with MLflow logging).
- `scripts/get_mojidata.ts`: download `moji.db` used as the IDS source database.
- `marimo/unsloth_ids.py`: exploratory notebook used for early experiments.
- `scripts/merge_gguf_qwen3vl.py`: merge LoRA-updated text weights into a base
  Qwen3-VL GGUF while preserving vision tensors.
- `docs/ollama_model_conversion.md`: step-by-step instructions to create an
  Ollama model from the merged GGUF.
- `docs/ignored_artifacts.md`: how to fetch gitignored assets needed to run
  the scripts.

## Typical workflow

1. Prepare the dataset (renders glyph images + normalized IDS):
   `python scripts/ids_experiment.py prepare`
2. Train a LoRA adapter with MLflow tracking:
   `python scripts/ids_experiment.py train --run-name <name>`
3. Evaluate on validation or test splits:
   `python scripts/ids_experiment.py evaluate --adapter-path <path>`
4. (Optional) Merge and register with Ollama using the docs.

## Outputs

- `outputs/ids_dataset`: rendered images and `train/val/test` JSONL splits.
- `outputs/IDS-LoRA-Qwen3-VL-2B*`: training runs and checkpoints.
- `outputs/merged-*.gguf` + `outputs/*.Modelfile`: Ollama-ready artifacts.
