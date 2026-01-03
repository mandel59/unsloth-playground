#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPT = (
    "Break down the Hanzi of the image into Ideographic Description Sequence (IDS). "
    "Output only the IDS."
)


def normalize_host(host: str) -> str:
    host = (host or "").strip()
    if not host:
        host = "http://127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def call_generate(host: str, payload: dict) -> dict:
    req = Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_sample_image(dataset_dir: Path) -> Path | None:
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        return None
    for root, _, files in os.walk(images_dir):
        for name in sorted(files):
            if name.lower().endswith(".png"):
                return Path(root) / name
    return None


def run_text_test(host: str, model: str, prompt: str) -> str:
    response = call_generate(
        host,
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
    )
    return response.get("response", "").strip()


def run_image_test(host: str, model: str, prompt: str, image_path: Path) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = call_generate(
        host,
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        },
    )
    return response.get("response", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Ollama IDS model.")
    parser.add_argument("--model", default="qwen3vl-ids")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", ""))
    parser.add_argument("--text", default="hello")
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dataset-dir", default="outputs/ids_dataset")
    args = parser.parse_args()

    host = normalize_host(args.host)
    image_path = Path(args.image) if args.image else find_sample_image(
        Path(args.dataset_dir)
    )

    try:
        text_out = run_text_test(host, args.model, args.text)
        print(f"text: {text_out}")
        if image_path:
            image_out = run_image_test(host, args.model, args.prompt, image_path)
            print(f"image: {image_path} -> {image_out}")
        else:
            print("image: skipped (no image found)")
    except (HTTPError, URLError) as exc:
        print(f"ollama request failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
