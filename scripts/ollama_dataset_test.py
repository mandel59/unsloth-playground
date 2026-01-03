#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
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


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_image_path(path: str | None, dataset_file: Path) -> Path | None:
    if not path:
        return None
    image_path = Path(path)
    if image_path.exists():
        return image_path
    candidate = dataset_file.parent / image_path
    if candidate.exists():
        return candidate
    return None


def call_generate(host: str, payload: dict) -> dict:
    req = Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_prediction(text: str, expected_char: str | None) -> str:
    text = (text or "").strip()
    if "=" in text:
        return text.split("=", 1)[1].strip()
    if expected_char and text.startswith(expected_char):
        return text[len(expected_char) :].lstrip(" =")
    if text.lower().startswith("ids"):
        return text[3:].lstrip(" :=")
    return text


def iter_samples(records: list[dict], max_samples: int, seed: int | None) -> Iterable[dict]:
    if max_samples <= 0 or max_samples >= len(records):
        return records
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return [records[i] for i in indices[:max_samples]]


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


@dataclass(frozen=True)
class SampleResult:
    image_path: Path
    expected: str
    predicted: str
    ok: bool


def evaluate_sample(
    host: str,
    model: str,
    prompt: str,
    image_path: Path,
    expected: str,
    expected_char: str | None,
) -> SampleResult:
    raw = run_image_test(host, model, prompt, image_path)
    pred = parse_prediction(raw, expected_char)
    return SampleResult(
        image_path=image_path,
        expected=expected,
        predicted=pred,
        ok=pred == expected,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama IDS model on outputs/ids_dataset/test.jsonl."
    )
    parser.add_argument("--model", default="qwen3vl-ids")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", ""))
    parser.add_argument(
        "--dataset-file", default="outputs/ids_dataset/test.jsonl"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--errors-only", action="store_true")
    args = parser.parse_args()

    host = normalize_host(args.host)
    dataset_file = Path(args.dataset_file)
    records = load_jsonl(dataset_file)
    if not records:
        print(f"no records in {dataset_file}", file=sys.stderr)
        return 1

    missing_images = 0
    results: list[SampleResult] = []

    try:
        items: list[tuple[Path, str, str | None]] = []
        for record in iter_samples(records, args.max_samples, args.seed):
            image_path = resolve_image_path(record.get("image_path"), dataset_file)
            if not image_path:
                missing_images += 1
                continue
            items.append(
                (
                    image_path,
                    record.get("ids", ""),
                    record.get("char"),
                )
            )

        if args.workers <= 1:
            for image_path, expected, expected_char in items:
                result = evaluate_sample(
                    host,
                    args.model,
                    args.prompt,
                    image_path,
                    expected,
                    expected_char,
                )
                results.append(result)
                if args.verbose or (args.errors_only and not result.ok):
                    status = "ok" if result.ok else "ng"
                    print(
                        f"{status}: {result.image_path} expected={result.expected} "
                        f"predicted={result.predicted}"
                    )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_sample,
                        host,
                        args.model,
                        args.prompt,
                        image_path,
                        expected,
                        expected_char,
                    )
                    for image_path, expected, expected_char in items
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if args.verbose or (args.errors_only and not result.ok):
                        status = "ok" if result.ok else "ng"
                        print(
                            f"{status}: {result.image_path} expected={result.expected} "
                            f"predicted={result.predicted}"
                        )
    except (HTTPError, URLError) as exc:
        print(f"ollama request failed: {exc}", file=sys.stderr)
        return 1

    total = len(results)
    correct = sum(1 for result in results if result.ok)
    accuracy = (correct / total) if total else 0.0
    print(
        f"summary: total={total} correct={correct} "
        f"accuracy={accuracy:.4f} missing_images={missing_images}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
