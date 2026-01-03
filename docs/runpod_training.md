# Runpod training plan (runpodctl)

This plan uses `runpodctl` for pod lifecycle management, `git clone` for the
repo, and in-pod downloads for the gitignored artifacts (`moji.db`, `fonts/`).

## Key assumptions

- The image is available in a registry: `mandel59/unsloth-playground:latest`.
- The container already includes the Python venv at:
  `/workspace/unsloth-playground/.venv`.
- Do not mount a volume over `/workspace`, or the venv will be hidden.
- `runpodctl create pod --help` shows `--volumePath` defaults to `/runpod`.
  When you create a volume (`--volumeSize` or `--networkVolumeId`), it is
  mounted there. If you change `--volumePath`, use that path instead.

## 1) Configure runpodctl

If your API key is already configured, skip the first command. To confirm,
`runpodctl get pod` should return a table (even if empty) without auth errors.

```
runpodctl config --apiKey "$RUNPOD_API_KEY"
runpodctl get cloud --mem 20 --vcpu 4
```

Pick a GPU type from `runpodctl get cloud` output (for example
`"NVIDIA GeForce RTX 4090"`).

## 2) Create a pod

```
runpodctl create pod \
  --name unsloth-ids-train \
  --imageName mandel59/unsloth-playground:latest \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --volumeSize 100 \
  --volumePath /runpod \
  --containerDiskSize 20
```

Optional: add `--env HF_TOKEN=...` to allow Hugging Face downloads.

## 3) Connect to the pod

If your SSH key is already registered with Runpod, skip the add-key step. To
confirm, run `runpodctl ssh list-keys` and check for your key comment or
fingerprint.

```
runpodctl get pod
runpodctl get pod <POD_ID>
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
```

Use the Runpod UI (or the connection details shown by `runpodctl get pod`) to
SSH into the pod.

## 4) Clone the repo onto the volume

```
cd /runpod
git clone https://github.com/mandel59/unsloth-playground.git
cd /runpod/unsloth-playground
git checkout <COMMIT_OR_TAG>
```

## 5) Download required artifacts in the pod

```
cd /runpod/unsloth-playground
deno run --allow-write=moji.db --unstable-raw-imports scripts/get_mojidata.ts
bash scripts/get_jigmo_fonts.sh
```

If you must avoid network downloads inside the pod, copy `moji.db` and
`fonts/` via SSH or by using `runpodctl send`/`runpodctl receive` after
installing `runpodctl` in the pod.

## 6) Prepare dataset (optional if already created)

```
/workspace/unsloth-playground/.venv/bin/python scripts/ids_experiment.py prepare \
  --db-path moji.db \
  --output-dir outputs/ids_dataset
```

## 7) Download the resume checkpoint from Hugging Face

The repo `mandel59/ids-lora-qwen3vl-2b-checkpoint` stores the checkpoint files
at the repo root (optimizer and trainer state included).

```
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mandel59/ids-lora-qwen3vl-2b-checkpoint",
    local_dir="outputs/ids-lora-qwen3vl-2b-checkpoint",
)
PY
```

This creates `outputs/ids-lora-qwen3vl-2b-checkpoint/` with the checkpoint files.

## 8) Train (resume from the checkpoint)

```
/workspace/unsloth-playground/.venv/bin/python scripts/ids_experiment.py train \
  --dataset-dir outputs/ids_dataset \
  --num-train-epochs 9 \
  --r 32 \
  --lora-alpha 32 \
  --max-length 192 \
  --run-name ids-lora-r32a32-runpod \
  --output-dir outputs/IDS-LoRA-Qwen3-VL-2B-r32a32-runpod \
  --resume-from-checkpoint outputs/ids-lora-qwen3vl-2b-checkpoint
```

For a quick smoke test, reduce data and epochs (see
`docs/ignored_artifacts.md` for the dataset flow).

## 9) Collect outputs and shut down

- Keep outputs under `/runpod/unsloth-playground/outputs` to persist on the
  volume.
- Download results via SSH or by attaching the volume to another pod. Example
  (SSH host/port/user are shown in the Runpod UI or `runpodctl get pod`):

```
# On the pod
cd /runpod/unsloth-playground
tar -czf /runpod/IDS-LoRA-Qwen3-VL-2B-r32a32-runpod.tgz \
  outputs/IDS-LoRA-Qwen3-VL-2B-r32a32-runpod

# On your local machine
scp -P <SSH_PORT> <USER>@<HOST>:/runpod/IDS-LoRA-Qwen3-VL-2B-r32a32-runpod.tgz .
tar -xzf IDS-LoRA-Qwen3-VL-2B-r32a32-runpod.tgz
```
- Stop or remove the pod:

```
runpodctl stop pod <POD_ID>
runpodctl remove pod <POD_ID>
```
