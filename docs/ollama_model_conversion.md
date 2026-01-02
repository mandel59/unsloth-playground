# Ollama model conversion (Qwen3-VL + IDS LoRA)

This project converts a Qwen3-VL LoRA adapter into an Ollama model by merging
text weights into the base GGUF while preserving the vision tensors.

## Prerequisites

- Base Qwen3-VL GGUF from Ollama (already working): `qwen3-vl:2b-instruct`
- LoRA merged HF model already exported to GGUF (text-only)
- Python venv with gguf-py available (this repo uses `tools/llama.cpp/gguf-py`)

## Files produced by this flow

- Combined GGUF: `outputs/merged-qwen3-vl-2b-ids-combined-fixed.gguf`
- Modelfile: `outputs/merged-qwen3-vl-2b-ids-combined-fixed.Modelfile`
- Ollama model name: `qwen3vl-ids-r32a32-combined-fixed`

## Step 1: Merge GGUFs (keep vision tensors)

The base GGUF contains all vision tensors. The text-only GGUF contains the
fine-tuned language weights. We merge by replacing only overlapping text
tensors while keeping all vision tensors and base metadata.

```
.venv\Scripts\python.exe scripts\merge_gguf_qwen3vl.py ^
  --base-gguf C:\Users\mande\.ollama\models\blobs\sha256-aafed9e48b157ae913cee994e0d9ac927af51e256feafbd923bf2852e8856d00 ^
  --text-gguf outputs\merged-qwen3-vl-2b-ids-full.gguf ^
  --output-gguf outputs\merged-qwen3-vl-2b-ids-combined-fixed.gguf
```

Notes:
- `scripts/merge_gguf_qwen3vl.py` writes tensor shapes using `data.shape`
  to avoid the GGUF writer’s internal shape reversal.
- The script skips `GGUF.*` KV keys to avoid duplicated headers.

## Step 2: Create the Modelfile

```
FROM C:/Users/mande/ws/unsloth-playground/outputs/merged-qwen3-vl-2b-ids-combined-fixed.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3-vl-instruct
PARSER qwen3-vl-instruct
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95
```

Save as: `outputs/merged-qwen3-vl-2b-ids-combined-fixed.Modelfile`

## Step 3: Register with Ollama

```
ollama create qwen3vl-ids-r32a32-combined-fixed -f outputs\merged-qwen3-vl-2b-ids-combined-fixed.Modelfile
```

## Step 4: Smoke test

```
ollama run qwen3vl-ids-r32a32-combined-fixed "hello"
```

## Step 5: Visual IDS inference example

```
ollama run qwen3vl-ids-r32a32-combined-fixed --image path\to\sample.png ^
  "Break down the Hanzi of the image into Ideographic Description Sequence (IDS). Output only the IDS."
```

## Troubleshooting

- `loras are not yet implemented`: Ollama does not load LoRA adapters directly.
  Use the GGUF merge flow above.
- `GGML_ASSERT((OD > 0) && "b too small compared to a") failed` or a Conv3D
  panic: indicates invalid vision tensor shapes (often from incorrect GGUF
  tensor shape writing). Rebuild the combined GGUF with the merge script.
- `unknown parameter 'PROJECTOR'` / `PARAMETER mmproj`: Ollama Modelfile does not
  support a separate mmproj GGUF. Vision must be inside the main GGUF.
