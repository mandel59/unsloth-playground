# Ollama model conversion (Qwen3-VL + IDS LoRA)

This project converts a Qwen3-VL LoRA adapter into an Ollama model by merging
text weights into the base GGUF while preserving the vision tensors.

## Prerequisites

- Base Qwen3-VL GGUF from Ollama (already working): `qwen3-vl:2b-instruct`
- LoRA merged HF model already exported to GGUF (text-only)
- Python venv with gguf-py available (this repo uses `tools/llama.cpp/gguf-py`)

## Files produced by this flow

- Text-only GGUF (from merged HF): `outputs/merged-qwen3-vl-2b-ids-text.gguf`
- Combined GGUF: `outputs/merged-qwen3-vl-2b-ids-combined.gguf`
- Modelfile: `outputs/merged-qwen3-vl-2b-ids-combined.Modelfile`
- Ollama model name: `qwen3vl-ids`

## Step 0: Build the text-only GGUF

First merge the LoRA adapter into the base HF model, then convert to a
text-only GGUF.

### 0a) Merge adapter into HF weights

```
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

base_dir = Path("PATH_TO_QWEN3_VL_2B_INSTRUCT_SNAPSHOT")
adapter_dir = Path("PATH_TO_ADAPTER_CHECKPOINT")
out_dir = Path("outputs/merged-qwen3-vl-2b-ids-text-hf")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_dir, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
tokenizer.save_pretrained(out_dir)
processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
processor.save_pretrained(out_dir)
```

### 0b) Convert merged HF to GGUF

```
.venv\Scripts\python.exe tools\llama.cpp\convert_hf_to_gguf.py ^
  outputs\merged-qwen3-vl-2b-ids-text-hf ^
  --outfile outputs\merged-qwen3-vl-2b-ids-text.gguf ^
  --outtype f16
```

## Step 1: Merge GGUFs (keep vision tensors)

The base GGUF contains all vision tensors. The text-only GGUF contains the
fine-tuned language weights. We merge by replacing only overlapping text
tensors while keeping all vision tensors and base metadata.

```
.venv\Scripts\python.exe scripts\merge_gguf_qwen3vl.py ^
  --base-gguf C:\Users\mande\.ollama\models\blobs\sha256-aafed9e48b157ae913cee994e0d9ac927af51e256feafbd923bf2852e8856d00 ^
  --text-gguf outputs\merged-qwen3-vl-2b-ids-full.gguf ^
  --output-gguf outputs\merged-qwen3-vl-2b-ids-combined.gguf
```

Notes:
- `scripts/merge_gguf_qwen3vl.py` writes tensor shapes using `data.shape`
  to avoid the GGUF writer’s internal shape reversal.
- The script skips `GGUF.*` KV keys to avoid duplicated headers.

## Step 2: Create the Modelfile

```
FROM C:/Users/mande/ws/unsloth-playground/outputs/merged-qwen3-vl-2b-ids-combined.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3-vl-instruct
PARSER qwen3-vl-instruct
PARAMETER temperature 0
PARAMETER num_predict 96
PARAMETER stop "\n"
```

Save as: `outputs/merged-qwen3-vl-2b-ids-combined.Modelfile`

## Step 3: Register with Ollama

```
ollama create qwen3vl-ids -f outputs\merged-qwen3-vl-2b-ids-combined.Modelfile
```

## Step 4: Smoke test

```
ollama run qwen3vl-ids "hello"
```

## Step 5: Visual IDS inference example (HTTP API)

Ollama's CLI does not accept `--image`. Use the HTTP API with base64-encoded
images.

```powershell
$imgPath = "path\\to\\sample.png"
$bytes = [IO.File]::ReadAllBytes($imgPath)
$b64 = [Convert]::ToBase64String($bytes)
$body = @{
  model = "qwen3vl-ids"
  prompt = "Break down the Hanzi of the image into Ideographic Description Sequence (IDS). Output only the IDS."
  images = @($b64)
  stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate `
  -Method Post -Body $body -ContentType "application/json"
```

## Troubleshooting

- `loras are not yet implemented`: Ollama does not load LoRA adapters directly.
  Use the GGUF merge flow above.
- `GGML_ASSERT((OD > 0) && "b too small compared to a") failed` or a Conv3D
  panic: indicates invalid vision tensor shapes (often from incorrect GGUF
  tensor shape writing). Rebuild the combined GGUF with the merge script.
- `unknown parameter 'PROJECTOR'` / `PARAMETER mmproj`: Ollama Modelfile does not
  support a separate mmproj GGUF. Vision must be inside the main GGUF.
