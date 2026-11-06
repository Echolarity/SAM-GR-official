"""Re-embed item titles only (description truncated to zero).

Used for the ``truncate-description`` intervention.  The embedding recipe is
identical to the pipeline's ``text_only`` mode (Qwen3-VL-8B, mean-pooled last
hidden state, title+description truncated to 1500 chars), except the
description is dropped.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "/data02/sht/SAM-GR"
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
CKPT = "/data02/qy/models/Qwen3-VL-8B-Instruct"
OUT = os.path.join(DATA, "Sports_and_Outdoors.emb-Qwen3VL-text_title.npy")
DEV = os.environ.get("DEV", "cuda:1")
BS = int(os.environ.get("BS", "8"))


def clean_text(text):
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


def text_message(text):
    return [{"role": "user", "content": [{"type": "text", "text": text[:1500]}]}]


def main():
    items = json.load(open(os.path.join(DATA, "Sports_and_Outdoors.item.json")))
    keys = sorted(items.keys(), key=lambda x: int(x))
    texts = []
    for k in keys:
        title = clean_text(items[k].get("title", ""))
        texts.append(title if title else "unknown item")
    print(f"{len(texts)} items; device={DEV}; bs={BS}", flush=True)

    processor = AutoProcessor.from_pretrained(CKPT)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, device_map={"": DEV}
    )
    model.eval()

    out = np.zeros((len(texts), 4096), dtype=np.float32)
    with torch.no_grad():
        for st in tqdm(range(0, len(texts), BS), desc="title-only"):
            batch = texts[st : st + BS]
            prompts = [processor.apply_chat_template(text_message(t), tokenize=False,
                                                     add_generation_prompt=False) for t in batch]
            inputs = processor(text=prompts, padding=True, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            last = outputs.hidden_states[-1]
            mask = inputs.attention_mask.unsqueeze(-1).expand(last.size()).float()
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out[st : st + len(batch)] = pooled.float().cpu().numpy()
    np.save(OUT, out)
    print("saved", OUT, out.shape, flush=True)


if __name__ == "__main__":
    main()
