import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "llama.cpp" / "gguf-py"))

import gguf


def copy_fields(writer: gguf.GGUFWriter, reader: gguf.GGUFReader) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        types = field.types
        value = field.contents()
        if types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, value, types[0], sub_type=types[1])
        else:
            writer.add_key_value(key, value, types[0])


def add_tensor(
    writer: gguf.GGUFWriter,
    tensor: gguf.gguf_reader.ReaderTensor,
) -> None:
    data = tensor.data
    raw_dtype = tensor.tensor_type if tensor.tensor_type not in (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
    ) else None
    raw_shape = data.shape
    writer.add_tensor(
        tensor.name,
        data,
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge GGUF text weights into a base Qwen3-VL GGUF (keeps vision tensors)."
    )
    parser.add_argument("--base-gguf", required=True, type=Path)
    parser.add_argument("--text-gguf", required=True, type=Path)
    parser.add_argument("--output-gguf", required=True, type=Path)
    args = parser.parse_args()

    base_reader = gguf.GGUFReader(str(args.base_gguf))
    text_reader = gguf.GGUFReader(str(args.text_gguf))
    text_tensors = {t.name: t for t in text_reader.tensors}

    writer = gguf.GGUFWriter(
        str(args.output_gguf),
        base_reader.get_field("general.architecture").contents(),
        use_temp_file=True,
    )
    if (alignment_field := base_reader.get_field("general.alignment")) is not None:
        writer.add_custom_alignment(int(alignment_field.contents()))

    copy_fields(writer, base_reader)

    for tensor in base_reader.tensors:
        source = text_tensors.get(tensor.name, tensor)
        add_tensor(writer, source)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
