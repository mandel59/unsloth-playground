import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    lora_model_name = "IDS-LoRA-Qwen3-VL-2B"
    model_name = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    return lora_model_name, model_name


@app.cell
def _():
    image_size = 128
    font_size = 120
    offset = (
        (image_size - font_size) / 2.0,
        (image_size - font_size) / 2.0 - font_size / 10.0,
    )
    return font_size, image_size, offset


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from unsloth import FastLanguageModel, FastVisionModel
    return (FastVisionModel,)


@app.cell
def _():
    import torch
    return


@app.cell
def _(FastVisionModel, model_name):
    source_model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return source_model, tokenizer


@app.cell
def _(source_model):
    source_model
    return


@app.cell
def _(FastVisionModel, source_model):
    model = FastVisionModel.get_peft_model(
        model=source_model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )
    return (model,)


@app.function
def load_ids():
    import sqlite3
    import polars as pl

    con = sqlite3.connect("file:moji.db?mode=ro", uri=True)
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                ucs AS char,
                CASE
                	WHEN ids = '？' THEN ucs
                	WHEN ids LIKE '〾%' THEN ucs
                	WHEN ids LIKE '⿻%' THEN ucs
                	ELSE ids
                END AS text
            FROM ids
            WHERE
                source LIKE '%J%'
                AND ids NOT LIKE '_%〾%'
                AND ids NOT LIKE '_%⿻%'
            ORDER BY ucs
        """
        )
        data = pl.from_dicts(cur, ["char", "text"])
    finally:
        con.close()
    return data


@app.cell
def _():
    ids = load_ids()
    return (ids,)


@app.cell
def _(ids):
    ids
    return


@app.cell
def _():
    from PIL import Image, ImageDraw, ImageFont
    return (ImageFont,)


@app.cell
def _():
    from fontTools.ttLib import TTFont
    return (TTFont,)


@app.cell
def _():
    import os
    return (os,)


@app.cell
def _(os):
    jigumo_fonts_path = [
        os.path.join("fonts", file)
        for file in ["Jigmo.ttf", "Jigmo2.ttf", "Jigmo3.ttf"]
    ]
    return (jigumo_fonts_path,)


@app.cell
def _(ImageFont, font_size, jigumo_fonts_path):
    jigumo_fonts_pil = [
        ImageFont.truetype(path, font_size) for path in jigumo_fonts_path
    ]
    return (jigumo_fonts_pil,)


@app.cell
def _(TTFont, jigumo_fonts_path):
    jigumo_fonts_tt = [TTFont(path) for path in jigumo_fonts_path]
    return (jigumo_fonts_tt,)


@app.function
def has_glyph(font, char):
    codepoint = ord(char)
    for table in font["cmap"].tables:
        if codepoint in table.cmap.keys():
            return True
    return False


@app.function
def find_font(fonts_tt, fonts_pil, char):
    for i, font in enumerate(fonts_tt):
        if has_glyph(font, char):
            return fonts_pil[i]
    return None


@app.cell
def _(image_size, offset):
    def char_to_image(fonts_tt, fonts_pil, char):
        if char is None:
            return None
        from PIL import Image, ImageDraw, ImageFont

        font = find_font(fonts_tt, fonts_pil, char)
        if not font:
            return None
        im = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.text(offset, char, fill=(0, 0, 0), font=font)
        return im
    return (char_to_image,)


@app.cell
def _(char_to_image, jigumo_fonts_pil, jigumo_fonts_tt):
    def char_to_image_jigumo(char):
        return char_to_image(jigumo_fonts_tt, jigumo_fonts_pil, char)
    return (char_to_image_jigumo,)


@app.cell
def _(char_to_image_jigumo, mo):
    mo.hstack(
        [char_to_image_jigumo(c) for c in ["漢", "𫙹", "\U00030123"]],
        justify="start",
    )
    return


@app.cell
def _():
    import polars as pl
    return


@app.cell
def _(char_to_image_jigumo, ids):
    dataset = ids.with_columns(
        ids["char"].map_elements(lambda c: char_to_image_jigumo(c)).alias("image")
    )
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.function
def convert_to_conversation(
    sample,
    instruction="Break down the Hanzi of the image into Ideographic Description Sequence (IDS).",
):
    char = sample["char"]
    text = sample["text"]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"{char}={text}"}],
        },
    ]
    return {"messages": conversation}


@app.cell
def _(dataset):
    converted_dataset = [
        convert_to_conversation(sample) for sample in dataset.iter_rows(named=True)
    ]
    return (converted_dataset,)


@app.cell
def _(converted_dataset):
    _d = converted_dataset[42]
    _d
    return


@app.cell
def _(FastVisionModel, model, tokenizer):
    def generate(image, messages, model=model):
        FastVisionModel.for_inference(model)  # Enable for inference!

        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        from transformers import TextStreamer

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False
        )
    return (generate,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=32,
                gradient_accumulation_steps=32,
                warmup_steps=3,
                # max_steps=30,
                num_train_epochs=1,  # Set this instead of max_steps for full training runs
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=os.path.join("outputs", lora_model_name),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train_2():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=32,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                # max_steps=30,
                num_train_epochs=1,  # Set this instead of max_steps for full training runs
                learning_rate=5e-5,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=os.path.join("outputs", lora_model_name + "_2"),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train_2,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train_3():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                # max_steps=30,
                num_train_epochs=2,  # Set this instead of max_steps for full training runs
                learning_rate=2e-5,
                logging_steps=8,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3408,
                output_dir=os.path.join("outputs", lora_model_name + "_3"),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train_3,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train_4():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                # max_steps=30,
                num_train_epochs=4,  # Set this instead of max_steps for full training runs
                learning_rate=2e-5,
                logging_steps=32,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3408,
                output_dir=os.path.join("outputs", lora_model_name + "_4"),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train_4,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train_5():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                # max_steps=30,
                num_train_epochs=4,  # Set this instead of max_steps for full training runs
                learning_rate=2e-5,
                logging_steps=100,
                save_steps=2000,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3408,
                output_dir=os.path.join("outputs", lora_model_name + "_5"),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train_5,)


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    lora_model_name,
    model,
    os,
    tokenizer,
):
    def train_6():
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        FastVisionModel.for_training(model)  # Enable for training!

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                # max_steps=30,
                num_train_epochs=4,  # Set this instead of max_steps for full training runs
                learning_rate=2e-5,
                logging_steps=100,
                save_steps=8000,
                optim="adamw_8bit",
                weight_decay=0.0001,
                lr_scheduler_type="linear",
                seed=3408,
                output_dir=os.path.join("outputs", lora_model_name + "_6"),
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=256,
            ),
        )
        trainer_stats = trainer.train()
        return trainer_stats
    return (train_6,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="train")
    train_button
    return (train_button,)


@app.cell
def _(mo):
    train_2_button = mo.ui.run_button(label="train 2")
    train_2_button
    return (train_2_button,)


@app.cell
def _(mo):
    train_3_button = mo.ui.run_button(label="train 3")
    train_3_button
    return (train_3_button,)


@app.cell
def _(mo):
    train_4_button = mo.ui.run_button(label="train 4")
    train_4_button
    return (train_4_button,)


@app.cell
def _(mo):
    train_5_button = mo.ui.run_button(label="train 5")
    train_5_button
    return (train_5_button,)


@app.cell
def _(mo):
    train_6_button = mo.ui.run_button(label="train 6")
    train_6_button
    return (train_6_button,)


@app.cell
def _(mo, train, train_button):
    mo.stop(not train_button.value)
    train()
    return


@app.cell
def _(mo, train_2, train_2_button):
    mo.stop(not train_2_button.value)
    train_2()
    return


@app.cell
def _(mo, train_3, train_3_button):
    mo.stop(not train_3_button.value)
    train_3()
    return


@app.cell
def _(mo, train_4, train_4_button):
    mo.stop(not train_4_button.value)
    train_4()
    return


@app.cell
def _(mo, train_5, train_5_button):
    mo.stop(not train_5_button.value)
    train_5()
    return


@app.cell
def _(mo, train_6, train_6_button):
    mo.stop(not train_6_button.value)
    train_6()
    return


@app.cell
def _():
    test_chars = [
    
        # "折",
        # "㔁",
        # "漢",
        # "𫙹",
        # "𩵛",
        # "県",
        # "券",
        # "兼",
        # "命",
        # "翼",
        # "翅",
        # "鬼",
        # "慣",
        # "渋",
        # "\U00030320",
        # "\U00030322",
        # "\U00030324",
        # "\U00030326",
        # "\U00030328",
        # "\U0003032A",
        # "\U0003032C",
        # "\U0003032E",
        # "\U00030420",
        # "\U00030422",
        # "\U00030424",
        # "\U00030426",
        # "\U00030428",
        # "\U0003042A",
        # "\U0003042C",
        # "\U0003042E",
    ]
    return (test_chars,)


@app.cell
def _(test_chars):
    test_chars
    return


@app.cell
def _(mo):
    generate_button = mo.ui.run_button(label="generate")
    generate_button
    return (generate_button,)


@app.cell
def _(char_to_image_jigumo, generate, generate_button, mo, test_chars):
    mo.stop(not generate_button.value)
    for _c in test_chars:
        _image = char_to_image_jigumo(_c)
        generate(
            _image,
            convert_to_conversation({"image": _image, "char": _c, "text": ""})[
                "messages"
            ][0:-1],
        )
    return


@app.cell
def _(tokenizer):
    _t = "⿱⿰生生月"
    _x = tokenizer(text=_t)["input_ids"][0]
    print(_x)
    _y = tokenizer(text=[*_t])["input_ids"]
    print(_y)
    print(_x == sum(_y, []))
    return


@app.cell
def _(mo):
    save_lora_model_button = mo.ui.run_button(label="save lora model")
    save_lora_model_button
    return (save_lora_model_button,)


@app.cell
def _(lora_model_name, mo, model, os, save_lora_model_button, tokenizer):
    mo.stop(not save_lora_model_button.value)
    _adapter_path = os.path.join("adapters", lora_model_name)
    model.save_pretrained(_adapter_path)
    tokenizer.save_pretrained(_adapter_path)
    print("lora model saved")
    return


if __name__ == "__main__":
    app.run()
