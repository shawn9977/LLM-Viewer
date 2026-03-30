"""
csv_to_json.py
将 computational_complexity CSV 文件解析为结构化 JSON，
提取每层的类型、名称、FLOPs、参数量和张量形状。
"""

import csv
import json
import re
import sys
from pathlib import Path


def parse_tensor_list(blob_str: str) -> list[dict]:
    """
    解析 InputBlobs / OutputBlobs 字段，例如：
      'f32(1x3x224x224) f32(32x3x3x3)'
    返回:
      [{"dtype": "f32", "shape": [1,3,224,224], "bytes": 4, "numel": 150528}, ...]
    """
    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "i8": 1, "u8": 1}
    tensors = []
    for token in blob_str.strip().split():
        m = re.match(r"(\w+)\(([^)]+)\)", token)
        if not m:
            continue
        dtype = m.group(1)
        shape = [int(d) for d in m.group(2).split("x")]
        numel = 1
        for d in shape:
            numel *= d
        elem_bytes = dtype_bytes.get(dtype, 4)
        tensors.append({
            "dtype": dtype,
            "shape": shape,
            "numel": numel,
            "size_bytes": numel * elem_bytes,
        })
    return tensors


def parse_layer_params(param_str: str) -> dict:
    """解析 LayerParams 字段，例如 '[auto_pad: explicit; strides: (2xs2)]'"""
    result = {}
    param_str = param_str.strip("[]")
    for item in param_str.split(";"):
        item = item.strip()
        if ":" in item:
            k, v = item.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def csv_to_json(csv_path: str, json_path: str):
    layers = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_type = row["LayerType"].strip()
            layer_name = row["LayerName"].strip()

            gflops = float(row["GFLOPs"]) if row["GFLOPs"].strip() not in ("", "-1.0000") else None
            giops  = float(row["GIOPs"])  if row["GIOPs"].strip()  not in ("", "-1.0000") else None
            mparams = float(row["MParams"]) if row["MParams"].strip() else 0.0

            input_tensors  = parse_tensor_list(row.get("InputBlobs", ""))
            output_tensors = parse_tensor_list(row.get("OutputBlobs", ""))
            layer_params   = parse_layer_params(row.get("LayerParams", ""))

            layers.append({
                "layer_type":     layer_type,
                "layer_name":     layer_name,
                "gflops":         gflops,
                "giops":          giops,
                "mparams":        mparams,
                "layer_params":   layer_params,
                "input_tensors":  input_tensors,
                "output_tensors": output_tensors,
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"layers": layers}, f, indent=2, ensure_ascii=False)

    print(f"已生成 {json_path}，共 {len(layers)} 层")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "computational_complexity (13).csv"
    json_file = sys.argv[2] if len(sys.argv) > 2 else "model_layers.json"
    csv_to_json(csv_file, json_file)
