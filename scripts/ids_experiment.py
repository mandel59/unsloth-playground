from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IDC_ARITY = {
    "〾": 1,
    "⿰": 2,
    "⿱": 2,
    "⿲": 3,
    "⿳": 3,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
    "⿼": 2,
    "⿽": 2,
    "⿾": 1,
    "⿿": 1,
    "㇯": 2,
}

IDC_SET = set(IDC_ARITY)
QUESTION_TOKENS = {"?", "？"}
TOKEN_RE = re.compile(r"\{[0-9]+\}|.")
SOURCE_PRIORITY = ["J", "K", "T", "C"]

CJK_RANGES = [
    (0x2E80, 0x2FDF),  # Radicals
    (0x31C0, 0x31EF),  # Strokes
    (0x3400, 0x4DBF),  # Ext A
    (0x4E00, 0x9FFF),  # Unified Ideographs
    (0xF900, 0xFAFF),  # Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Ext B
    (0x2A700, 0x2B73F),  # Ext C
    (0x2B740, 0x2B81F),  # Ext D
    (0x2B820, 0x2CEAF),  # Ext E
    (0x2CEB0, 0x2EBEF),  # Ext F
    (0x2EBF0, 0x2EE5F),  # Ext I
    (0x30000, 0x3134F),  # Ext G
    (0x31350, 0x323AF),  # Ext H
    (0x323B0, 0x3347F),  # Ext J
]


@dataclass(frozen=True)
class IDSRecord:
    ucs: str
    ids: str
    source: str
    source_rank: int
    ids_len: int
    normalized_reason: str


def tokenize_ids(ids: str) -> list[str]:
    return TOKEN_RE.findall(ids)


def is_cjk_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    code = ord(ch)
    for start, end in CJK_RANGES:
        if start <= code <= end:
            return True
    return False


def find_unknown_tokens(ids: str) -> list[str]:
    unknown = []
    for tok in tokenize_ids(ids):
        if tok in IDC_SET:
            continue
        if tok in QUESTION_TOKENS:
            continue
        if tok.startswith("{") and tok.endswith("}") and tok[1:-1].isdigit():
            continue
        if is_cjk_char(tok):
            continue
        if tok.isspace():
            unknown.append(tok)
            continue
        unknown.append(tok)
    return unknown


def source_rank(source: str) -> int:
    source = source or ""
    for idx, key in enumerate(SOURCE_PRIORITY):
        if key in source:
            return idx
    return len(SOURCE_PRIORITY)


def normalize_ids(ucs: str, ids: str) -> tuple[str | None, str]:
    ids = (ids or "").strip()
    if not ids:
        return None, "empty"
    if ids in QUESTION_TOKENS:
        return ucs, "replaced_question"
    if ids.startswith("〾"):
        return ucs, "replaced_op_303e"
    if ids.startswith("⿻"):
        return ucs, "replaced_op_2ffb"
    if any(q in ids for q in QUESTION_TOKENS):
        return None, "contains_question"
    if "〾" in ids:
        return None, "contains_op_303e"
    if "⿻" in ids:
        return None, "contains_op_2ffb"
    return ids, "ok"


def is_valid_ids(ids: str) -> bool:
    tokens = tokenize_ids(ids)
    if not tokens:
        return False
    stack = [1]
    for idx, tok in enumerate(tokens):
        if tok.isspace():
            return False
        if not stack:
            return False
        stack[-1] -= 1
        if stack[-1] == 0:
            stack.pop()
        if tok in IDC_SET:
            stack.append(IDC_ARITY[tok])
        if not stack and idx != len(tokens) - 1:
            return False
    return len(stack) == 0


def ids_length(ids: str) -> int:
    return len(tokenize_ids(ids))


def iter_ids_rows(db_path: Path) -> Iterable[tuple[str, str, str]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute("SELECT ucs, ids, source FROM ids ORDER BY ucs")
        for row in cur:
            yield row
    finally:
        con.close()


def select_preferred_ids(
    rows: list[tuple[str, str, str]],
    stats: dict,
    unknown_tokens: Counter,
) -> IDSRecord | None:
    candidates: list[IDSRecord] = []
    for ucs, ids, source in rows:
        for tok in find_unknown_tokens(ids or ""):
            unknown_tokens[tok] += 1

        normalized, reason = normalize_ids(ucs, ids)
        stats["normalized_reasons"][reason] += 1
        if normalized is None:
            continue
        if not is_valid_ids(normalized):
            stats["invalid_grammar"] += 1
            continue

        record = IDSRecord(
            ucs=ucs,
            ids=normalized,
            source=source,
            source_rank=source_rank(source),
            ids_len=ids_length(normalized),
            normalized_reason=reason,
        )
        candidates.append(record)

    if not candidates:
        return None

    candidates.sort(key=lambda r: (r.source_rank, -r.ids_len, r.ids))
    best = candidates[0]
    stats["selected_source_rank"][best.source_rank] += 1
    if len(candidates) > 1:
        stats["multiple_candidates"] += 1
    return best


def load_font_assets(font_paths: list[Path], font_size: int):
    try:
        from PIL import ImageFont
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and fonttools are required for rendering."
        ) from exc

    fonts_pil = [ImageFont.truetype(str(path), size=font_size) for path in font_paths]
    fonts_tt = [TTFont(str(path)) for path in font_paths]
    return fonts_tt, fonts_pil


def has_glyph(font, char: str) -> bool:
    codepoint = ord(char)
    for table in font["cmap"].tables:
        if codepoint in table.cmap:
            return True
    return False


def find_font(fonts_tt, fonts_pil, char: str):
    for i, font in enumerate(fonts_tt):
        if has_glyph(font, char):
            return fonts_pil[i]
    return None


def render_char_image(
    fonts_tt,
    fonts_pil,
    char: str,
    image_size: int,
    font_size: int,
):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for rendering.") from exc

    font = find_font(fonts_tt, fonts_pil, char)
    if not font:
        return None
    offset = (
        (image_size - font_size) / 2.0,
        (image_size - font_size) / 2.0 - font_size / 10.0,
    )
    im = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text(offset, char, fill=(0, 0, 0), font=font)
    return im


def image_path_for_char(image_dir: Path, char: str) -> Path:
    hex_code = f"{ord(char):06X}"
    return image_dir / hex_code[:2] / hex_code[2:4] / f"{hex_code}.png"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_records(records: list[dict], seed: int, ratios: tuple[float, float, float]):
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def prepare_dataset(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    font_paths = [Path(p) for p in args.fonts]
    fonts_tt, fonts_pil = load_font_assets(font_paths, args.font_size)

    stats = {
        "total_rows": 0,
        "unique_chars": 0,
        "selected_chars": 0,
        "invalid_grammar": 0,
        "missing_glyphs": 0,
        "multiple_candidates": 0,
        "normalized_reasons": Counter(),
        "selected_source_rank": Counter(),
    }
    unknown_tokens = Counter()
    missing_glyphs = []

    records = []
    grouped_rows: list[tuple[str, str, str]] = []
    last_ucs = None

    for row in iter_ids_rows(db_path):
        stats["total_rows"] += 1
        ucs, ids, source = row
        if last_ucs is None:
            last_ucs = ucs
        if ucs != last_ucs:
            stats["unique_chars"] += 1
            selected = select_preferred_ids(grouped_rows, stats, unknown_tokens)
            if selected:
                image = render_char_image(
                    fonts_tt,
                    fonts_pil,
                    selected.ucs,
                    args.image_size,
                    args.font_size,
                )
                if image is None:
                    stats["missing_glyphs"] += 1
                    missing_glyphs.append(selected.ucs)
                else:
                    image_path = image_path_for_char(image_dir, selected.ucs)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                    records.append(
                        {
                            "char": selected.ucs,
                            "ids": selected.ids,
                            "source": selected.source,
                            "normalized_reason": selected.normalized_reason,
                            "image_path": str(image_path),
                        }
                    )
            grouped_rows = []
            last_ucs = ucs
        grouped_rows.append(row)

    if grouped_rows:
        stats["unique_chars"] += 1
        selected = select_preferred_ids(grouped_rows, stats, unknown_tokens)
        if selected:
            image = render_char_image(
                fonts_tt,
                fonts_pil,
                selected.ucs,
                args.image_size,
                args.font_size,
            )
            if image is None:
                stats["missing_glyphs"] += 1
                missing_glyphs.append(selected.ucs)
            else:
                image_path = image_path_for_char(image_dir, selected.ucs)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                records.append(
                    {
                        "char": selected.ucs,
                        "ids": selected.ids,
                        "source": selected.source,
                        "normalized_reason": selected.normalized_reason,
                        "image_path": str(image_path),
                    }
                )

    stats["selected_chars"] = len(records)

    train, val, test = split_records(
        records, args.seed, (args.train_ratio, args.val_ratio, args.test_ratio)
    )
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    stats_path = output_dir / "stats.json"
    unknown_path = output_dir / "unknown_tokens.json"
    missing_path = output_dir / "missing_glyphs.json"
    manifest_path = output_dir / "manifest.json"

    stats_serializable = {
        "total_rows": stats["total_rows"],
        "unique_chars": stats["unique_chars"],
        "selected_chars": stats["selected_chars"],
        "invalid_grammar": stats["invalid_grammar"],
        "missing_glyphs": stats["missing_glyphs"],
        "multiple_candidates": stats["multiple_candidates"],
        "normalized_reasons": dict(stats["normalized_reasons"]),
        "selected_source_rank": dict(stats["selected_source_rank"]),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
    with unknown_path.open("w", encoding="utf-8") as f:
        json.dump(unknown_tokens.most_common(), f, ensure_ascii=False, indent=2)
    with missing_path.open("w", encoding="utf-8") as f:
        json.dump(missing_glyphs, f, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "db_path": str(db_path),
                "fonts": [str(p) for p in font_paths],
                "image_size": args.image_size,
                "font_size": args.font_size,
                "source_priority": SOURCE_PRIORITY,
                "filters": {
                    "exclude_contains": ["?", "？", "〾", "⿻"],
                    "replace_if": ["ids == '?'", "ids == '？'", "ids startswith '〾'", "ids startswith '⿻'"],
                    "grammar_only": True,
                },
                "splits": {
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                    "seed": args.seed,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


class IDSMessageDataset:
    def __init__(self, records: list[dict], instruction: str):
        self.records = records
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["ids"]}],
            },
        ]
        return {"messages": messages}


def ids_difficulty(ids: str) -> int:
    tokens = tokenize_ids(ids)
    return len(tokens) + sum(tok in IDC_SET for tok in tokens)


def build_curriculum(records: list[dict]) -> list[tuple[str, list[dict]]]:
    scored = [(ids_difficulty(r["ids"]), r) for r in records]
    scored.sort(key=lambda t: t[0])
    total = len(scored)
    if total == 0:
        return []
    t1 = scored[int(total * 0.33)][0]
    t2 = scored[int(total * 0.66)][0]
    easy = [r for score, r in scored if score <= t1]
    mid = [r for score, r in scored if t1 < score <= t2]
    hard = [r for score, r in scored if score > t2]
    return [
        ("stage1_easy", easy),
        ("stage2_easy_mid", easy + mid),
        ("stage3_full", easy + mid + hard),
    ]


def train_model(args: argparse.Namespace) -> None:
    try:
        import mlflow
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise RuntimeError(
            "unsloth, trl, and mlflow are required for training."
        ) from exc

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    dataset_path = Path(args.dataset_dir) / f"{args.split}.jsonl"
    train_records = read_jsonl(dataset_path)
    eval_dataset = None
    eval_path = None
    if args.evaluation_strategy != "no":
        eval_path = Path(args.dataset_dir) / f"{args.eval_split}.jsonl"
        eval_records = read_jsonl(eval_path)
        if not eval_records:
            raise ValueError(f"Eval split is empty: {eval_path}")
        eval_dataset = IDSMessageDataset(eval_records, args.instruction)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "cfg_model_name": args.model_name,
                "cfg_adapter_name": args.adapter_name,
                "cfg_dataset_path": str(dataset_path),
                "cfg_split": args.split,
                "cfg_per_device_train_batch_size": args.per_device_train_batch_size,
                "cfg_gradient_accumulation_steps": args.gradient_accumulation_steps,
                "cfg_warmup_steps": args.warmup_steps,
                "cfg_num_train_epochs": args.num_train_epochs,
                "cfg_learning_rate": args.learning_rate,
                "cfg_weight_decay": args.weight_decay,
                "cfg_lr_scheduler_type": args.lr_scheduler_type,
                "cfg_seed": args.seed,
                "cfg_lora_r": args.r,
                "cfg_lora_alpha": args.lora_alpha,
                "cfg_lora_dropout": args.lora_dropout,
                "cfg_finetune_vision_layers": args.finetune_vision_layers,
                "cfg_finetune_language_layers": args.finetune_language_layers,
                "cfg_finetune_attention_modules": args.finetune_attention_modules,
                "cfg_finetune_mlp_modules": args.finetune_mlp_modules,
                "cfg_max_length": args.max_length,
                "cfg_curriculum": args.curriculum,
                "cfg_eval_split": args.eval_split,
                "cfg_evaluation_strategy": args.evaluation_strategy,
                "cfg_eval_steps": args.eval_steps,
                "cfg_save_strategy": args.save_strategy,
                "cfg_save_steps": args.save_steps,
                "cfg_save_total_limit": args.save_total_limit,
                "cfg_load_best_model_at_end": args.load_best_model_at_end,
                "cfg_metric_for_best_model": args.metric_for_best_model,
                "cfg_greater_is_better": args.greater_is_better,
                "cfg_resume_from_checkpoint": args.resume_from_checkpoint,
            }
        )

        source_model, tokenizer = FastVisionModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model=source_model,
            finetune_vision_layers=args.finetune_vision_layers,
            finetune_language_layers=args.finetune_language_layers,
            finetune_attention_modules=args.finetune_attention_modules,
            finetune_mlp_modules=args.finetune_mlp_modules,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
            use_gradient_checkpointing="unsloth",
        )

        FastVisionModel.for_training(model)
        base_output_dir = Path(args.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        load_best_model = args.load_best_model_at_end
        if args.evaluation_strategy == "no" or args.save_strategy == "no":
            load_best_model = False

        def run_stage(stage_name: str, stage_records: list[dict]) -> None:
            dataset = IDSMessageDataset(stage_records, args.instruction)
            stage_output_dir = base_output_dir
            if args.curriculum:
                stage_output_dir = base_output_dir / stage_name
                stage_output_dir.mkdir(parents=True, exist_ok=True)
            resume_path = None
            if args.resume_from_checkpoint:
                resume_candidate = Path(args.resume_from_checkpoint)
                if args.curriculum:
                    if stage_output_dir in resume_candidate.parents:
                        resume_path = str(resume_candidate)
                else:
                    resume_path = str(resume_candidate)
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                args=SFTConfig(
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    warmup_steps=args.warmup_steps,
                    num_train_epochs=args.num_train_epochs,
                    learning_rate=args.learning_rate,
                    logging_steps=args.logging_steps,
                    optim=args.optim,
                    weight_decay=args.weight_decay,
                    lr_scheduler_type=args.lr_scheduler_type,
                    seed=args.seed,
                    output_dir=str(stage_output_dir),
                    report_to=args.report_to,
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=args.max_length,
                    eval_strategy=args.evaluation_strategy,
                    eval_steps=args.eval_steps,
                    save_strategy=args.save_strategy,
                    save_steps=args.save_steps,
                    save_total_limit=args.save_total_limit,
                    load_best_model_at_end=load_best_model,
                    metric_for_best_model=args.metric_for_best_model,
                    greater_is_better=args.greater_is_better,
                ),
            )
            metrics = trainer.train(resume_from_checkpoint=resume_path).metrics
            metrics = {f"{stage_name}/{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(metrics)

        if args.curriculum:
            for stage_name, stage_records in build_curriculum(train_records):
                with mlflow.start_run(run_name=stage_name, nested=True):
                    mlflow.log_param("stage_name", stage_name)
                    mlflow.log_param("stage_samples", len(stage_records))
                    run_stage(stage_name, stage_records)
        else:
            run_stage("train", train_records)

        model.save_pretrained(base_output_dir / args.adapter_name)
        tokenizer.save_pretrained(base_output_dir / args.adapter_name)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def parse_prediction(text: str, expected_char: str) -> str:
    text = (text or "").strip()
    if "=" in text:
        return text.split("=", 1)[1].strip()
    if text.startswith(expected_char):
        return text[len(expected_char) :].lstrip(" =")
    if text.lower().startswith("ids"):
        return text[3:].lstrip(" :=")
    return text


def evaluate_model(args: argparse.Namespace) -> None:
    try:
        import mlflow
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "unsloth and mlflow are required for evaluation."
        ) from exc

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    dataset_path = Path(args.dataset_dir) / f"{args.split}.jsonl"
    records = read_jsonl(dataset_path)
    if args.max_samples and args.max_samples < len(records):
        records = records[: args.max_samples]

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "adapter_path": args.adapter_path,
                "dataset_path": str(dataset_path),
                "split": args.split,
                "max_samples": args.max_samples,
                "predictions_path": args.predictions_path,
                "errors_only": args.errors_only,
            }
        )

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=args.model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model.load_adapter(args.adapter_path)
        FastVisionModel.for_inference(model)

        correct = 0
        total = 0
        total_edit = 0
        grammar_ok = 0

        predictions_file = None
        if args.predictions_path:
            predictions_path = Path(args.predictions_path)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_file = predictions_path.open("w", encoding="utf-8")

        try:
            for record in records:
                from PIL import Image

                image = Image.open(record["image_path"]).convert("RGB")
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.instruction},
                            {"type": "image", "image": image},
                        ],
                    }
                ]
                input_text = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True
                )
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )
                input_len = inputs["input_ids"].shape[-1]
                decoded = tokenizer.decode(
                    output[0][input_len:], skip_special_tokens=True
                )
                pred_ids = parse_prediction(decoded, record["char"])
                total += 1
                if pred_ids == record["ids"]:
                    correct += 1
                edit_distance = levenshtein(pred_ids, record["ids"])
                total_edit += edit_distance
                pred_grammar_ok = is_valid_ids(pred_ids)
                if pred_grammar_ok:
                    grammar_ok += 1

                if predictions_file:
                    if not args.errors_only or pred_ids != record["ids"]:
                        predictions_file.write(
                            json.dumps(
                                {
                                    "char": record["char"],
                                    "ids": record["ids"],
                                    "pred_ids": pred_ids,
                                    "edit_distance": edit_distance,
                                    "grammar_ok": pred_grammar_ok,
                                    "image_path": record["image_path"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            if predictions_file:
                predictions_file.close()

        metrics = {
            "exact_match": correct / total if total else 0.0,
            "avg_edit_distance": total_edit / total if total else 0.0,
            "grammar_ok_rate": grammar_ok / total if total else 0.0,
        }
        mlflow.log_metrics(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IDS dataset preparation and Unsloth LoRA experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare IDS dataset.")
    prepare.add_argument("--db-path", default="moji.db")
    prepare.add_argument("--output-dir", default="outputs/ids_dataset")
    prepare.add_argument("--image-size", type=int, default=128)
    prepare.add_argument("--font-size", type=int, default=120)
    prepare.add_argument(
        "--fonts",
        nargs="+",
        default=[
            "fonts/Jigmo.ttf",
            "fonts/Jigmo2.ttf",
            "fonts/Jigmo3.ttf",
        ],
    )
    prepare.add_argument("--seed", type=int, default=3407)
    prepare.add_argument("--train-ratio", type=float, default=0.9)
    prepare.add_argument("--val-ratio", type=float, default=0.05)
    prepare.add_argument("--test-ratio", type=float, default=0.05)
    prepare.set_defaults(func=prepare_dataset)

    train = subparsers.add_parser("train", help="Train IDS LoRA adapter.")
    train.add_argument("--dataset-dir", default="outputs/ids_dataset")
    train.add_argument("--split", default="train")
    train.add_argument("--resume-from-checkpoint", default=None)
    train.add_argument("--eval-split", default="val")
    train.add_argument(
        "--evaluation-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    train.add_argument("--eval-steps", type=int, default=50)
    train.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    train.add_argument("--save-steps", type=int, default=50)
    train.add_argument("--save-total-limit", type=int, default=3)
    train.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--metric-for-best-model", default="eval_loss")
    train.add_argument(
        "--greater-is-better", action=argparse.BooleanOptionalAction, default=False
    )
    train.add_argument(
        "--model-name",
        default="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    )
    train.add_argument("--adapter-name", default="IDS-LoRA-Qwen3-VL-2B")
    train.add_argument("--output-dir", default="outputs/IDS-LoRA-Qwen3-VL-2B")
    train.add_argument("--experiment-name", default="ids-lora-qwen3-vl")
    train.add_argument("--run-name", default=None)
    train.add_argument("--mlflow-uri", default=None)
    train.add_argument(
        "--instruction",
        default="Break down the Hanzi of the image into Ideographic Description Sequence (IDS). Output only the IDS.",
    )
    train.add_argument("--curriculum", action="store_true")
    train.add_argument("--per-device-train-batch-size", type=int, default=32)
    train.add_argument("--gradient-accumulation-steps", type=int, default=32)
    train.add_argument("--warmup-steps", type=int, default=3)
    train.add_argument("--num-train-epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--logging-steps", type=int, default=1)
    train.add_argument("--optim", default="adamw_8bit")
    train.add_argument("--weight-decay", type=float, default=0.001)
    train.add_argument("--lr-scheduler-type", default="linear")
    train.add_argument("--seed", type=int, default=3407)
    train.add_argument("--r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--report-to", default="mlflow")
    train.add_argument(
        "--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False
    )
    train.add_argument(
        "--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument(
        "--finetune-attention-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True
    )
    train.set_defaults(func=train_model)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate IDS adapter.")
    evaluate.add_argument("--dataset-dir", default="outputs/ids_dataset")
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument(
        "--model-name",
        default="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
    )
    evaluate.add_argument("--adapter-path", required=True)
    evaluate.add_argument("--experiment-name", default="ids-lora-qwen3-vl")
    evaluate.add_argument("--run-name", default=None)
    evaluate.add_argument("--mlflow-uri", default=None)
    evaluate.add_argument(
        "--instruction",
        default="Break down the Hanzi of the image into Ideographic Description Sequence (IDS). Output only the IDS.",
    )
    evaluate.add_argument("--max-samples", type=int, default=0)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--predictions-path", default=None)
    evaluate.add_argument(
        "--errors-only", action=argparse.BooleanOptionalAction, default=False
    )
    evaluate.set_defaults(func=evaluate_model)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
