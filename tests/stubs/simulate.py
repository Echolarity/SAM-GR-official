"""Simulate expensive pipeline stages for the smoke test.

This is invoked by ``tests/stubs/python`` only for RQ-VAE, index generation and
evaluation commands. Real preprocessing and data possession are delegated to the
real interpreter by the wrapper script.
"""

import json
import os
import sys


REAL = os.environ.get("SMOKE_REAL_PYTHON", "/data/sht/conda/mimigenrec/bin/python")
ARGS = sys.argv[1:]
JOINED = " ".join(ARGS)


def opt(name, default=None):
    if name in ARGS:
        i = ARGS.index(name)
        if i + 1 < len(ARGS):
            return ARGS[i + 1]
    return default


def cuda_list():
    raw = opt("--cuda_list", "0")
    return raw.split()


def write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


if "rqvae.py" in JOINED:
    ckpt_dir = opt("--ckpt_dir", "rq_output/smoke")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "best_collision_model.pth"), "wb") as f:
        f.write(b"smoke-checkpoint")
    print(f"[stub] rqvae -> {ckpt_dir}")
    raise SystemExit(0)

if "generate_indices.py" in JOINED:
    dataset = opt("--dataset")
    out_dir = opt("--output_dir")
    item_path = os.path.join(out_dir, f"{dataset}.item.json")
    with open(item_path, encoding="utf-8") as f:
        items = json.load(f)
    index = {}
    for i, item_id in enumerate(items):
        index[str(item_id)] = [
            f"<a_{i % 8}>",
            f"<b_{i % 4}>",
            f"<c_{i % 2}>",
            f"<d_{i % 3}>",
        ]
    write_json(os.path.join(out_dir, f"{dataset}.index.json"), index)
    print(f"[stub] generate_indices -> {len(index)} items")
    raise SystemExit(0)

if "samgr.split_json" in JOINED:
    input_path = opt("--input_path")
    output_path = opt("--output_path")
    gpus = cuda_list()
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(output_path, exist_ok=True)
    n = max(len(gpus), 1)
    for i, gpu in enumerate(gpus or ["0"]):
        write_json(os.path.join(output_path, f"{gpu}.json"), data[i::n])
    print(f"[stub] split_json -> {len(gpus or ['0'])} parts")
    raise SystemExit(0)

if "samgr.merge_json" in JOINED:
    input_path = opt("--input_path")
    output_path = opt("--output_path")
    merged = []
    for gpu in cuda_list():
        part = os.path.join(input_path, f"{gpu}_result.json")
        if os.path.exists(part):
            with open(part, encoding="utf-8") as f:
                merged.extend(json.load(f))
    write_json(output_path, merged)
    print(f"[stub] merge_json -> {len(merged)} rows")
    raise SystemExit(0)

if "samgr.evaluate" in JOINED:
    result_path = opt("--result_json_path")
    if "--metrics_only" in JOINED:
        if result_path:
            write_json(os.path.join(os.path.dirname(result_path), "metrics.json"), {"stub": True})
        print("[stub] evaluate --metrics_only")
        raise SystemExit(0)
    if result_path:
        write_json(result_path, [])
        write_json(os.path.join(os.path.dirname(result_path), "metrics.json"), {"stub": True})
    print("[stub] evaluate")
    raise SystemExit(0)

os.execv(REAL, [REAL, *ARGS])
