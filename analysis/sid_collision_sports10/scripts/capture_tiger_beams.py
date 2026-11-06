import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = "/data02/sht/SAM-GR"
sys.path.insert(0, os.path.join(BASE, "scripts", "tiger"))
from train_tiger import SIDDataset, Tiger, beam_search, build_trie, build_vocab, collate, load_samples

DATA = os.path.join(BASE, "analysis", "sid_collision_sports10", "tiger_data_out")
OUT = os.path.join(BASE, "analysis", "sid_collision_sports10", "results", "beams")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(BASE, "experiments_tiger", "tiger_Sports_and_Outdoors.pt")


def main():
    base = os.path.join(DATA, "sft")
    train_rows = load_samples(os.path.join(base, "train.json"))
    valid_rows = load_samples(os.path.join(base, "valid.json"))
    test_rows = load_samples(os.path.join(base, "test.json"))
    tokens, tok2id = build_vocab([train_rows, valid_rows, test_rows])
    id2tok = {v: k for k, v in tok2id.items()}
    vocab = len(tokens) + 3
    test_ds = SIDDataset(test_rows, tok2id)
    trie = build_trie(train_rows + valid_rows + test_rows, tok2id)
    device = "cuda:0"
    model = Tiger(vocab).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)
    all_beams = []
    all_targets = []
    with torch.no_grad():
        for bi, (src, tgt) in enumerate(loader):
            cands = beam_search(model, src, trie, beams=50, device=device)
            for b in range(src.size(0)):
                beams = []
                for sc, pref in cands[b]:
                    beams.append("".join(id2tok[int(x)] for x in pref))
                all_beams.append(beams)
                all_targets.append("".join(id2tok[int(x)] for x in tgt[b].tolist()))
            if bi % 100 == 0:
                print(f"batch {bi}/{len(loader)}", flush=True)
    maxlen = max(len(x) for x in all_beams)
    arr = np.full((len(all_beams), maxlen), "", dtype=object)
    for i, row in enumerate(all_beams):
        arr[i, :len(row)] = row
    np.save(os.path.join(OUT, "tiger_beams.npy"), arr, allow_pickle=True)
    np.save(os.path.join(OUT, "tiger_targets.npy"), np.array(all_targets, dtype=object), allow_pickle=True)
    print("saved", arr.shape, len(all_targets))


if __name__ == "__main__":
    main()
