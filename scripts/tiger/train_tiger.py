#!/usr/bin/env python3
"""TIGER-style text-only baseline.

Architecture: Transformer encoder-decoder (T5-like) trained from scratch on the
RQ-VAE text-only semantic IDs, with prefix-tree (Trie) constrained beam search.
(T5 pretrained weights are unavailable offline, so we train the same architecture
from scratch and report it as "TIGER (from scratch)".)

Data: data/<CAT>/sft/{train,valid,test}.json produced by samgr/preprocess.py
      (input contains the history SID sequence, output is the target SID).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

SID_RE = re.compile(r"<a_\d+><b_\d+><c_\d+><d_\d+>")
TOK_RE = re.compile(r"<[abcd]_\d+>")

PAD, BOS, EOS = 0, 1, 2


def load_samples(path):
    samples = json.load(open(path))
    out = []
    for s in samples:
        tgt = str(s.get("output", "")).strip()
        hist = SID_RE.findall(str(s.get("input", "")))
        if not hist or len(TOK_RE.findall(tgt)) != 4:
            continue
        if tgt not in hist:  # keep all (do not leak target into input)
            out.append((hist, tgt))
    return out


def sid_tokens(sid):
    return TOK_RE.findall(sid)


def build_vocab(splits):
    toks = set()
    for rows in splits:
        for hist, tgt in rows:
            for sid in hist:
                toks.update(sid_tokens(sid))
            toks.update(sid_tokens(tgt))
    tokens = sorted(toks)
    tok2id = {t: i + 3 for i, t in enumerate(tokens)}
    return tokens, tok2id


class SIDDataset(Dataset):
    def __init__(self, rows, tok2id, max_src=200, max_items=50):
        self.items = []
        for hist, tgt in rows:
            src = []
            for sid in hist[-max_items:]:
                src += [tok2id[t] for t in sid_tokens(sid)]
            src = src[-max_src:]
            t = [tok2id[t] for t in sid_tokens(tgt)]
            if len(src) == 0 or len(t) != 4:
                continue
            self.items.append((src, t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    src = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    tgt = torch.tensor([x[1] for x in batch], dtype=torch.long)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD)
    return src, tgt


class Tiger(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=8, enc_layers=4, dec_layers=4,
                 dim_ff=1024, dropout=0.1, max_len=256):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.tr = nn.Transformer(d_model=d_model, nhead=nhead,
                                 num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
                                 dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def encode(self, src):
        L = src.size(1)
        x = self.emb(src) + self.pos(torch.arange(L, device=src.device))[None]
        return self.tr.encoder(x, src_key_padding_mask=(src == PAD))

    def decode(self, memory, tgt, mem_pad_mask):
        L = tgt.size(1)
        x = self.emb(tgt) + self.pos(torch.arange(L, device=tgt.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        out = self.tr.decoder(x, memory, tgt_mask=mask, memory_key_padding_mask=mem_pad_mask)
        return self.fc(out)

    def forward(self, src, tgt):
        mem = self.encode(src)
        inp = torch.cat([torch.full((tgt.size(0), 1), BOS, device=tgt.device), tgt[:, :-1]], dim=1)
        return self.decode(mem, inp, src == PAD)


def build_trie(rows, tok2id):
    trie = defaultdict(set)
    for _, tgt in rows:
        ids = [tok2id[t] for t in sid_tokens(tgt)]
        if len(ids) != 4:
            continue
        for k in range(4):
            trie[tuple(ids[:k])].add(ids[k])
    return trie


@torch.no_grad()
def beam_search(model, src, trie, beams=50, device="cuda:0"):
    model.eval()
    B = src.size(0)
    src = src.to(device)
    mem = model.encode(src)
    pad = (src == PAD)
    cands = [[(0.0, ())] for _ in range(B)]
    for step in range(4):
        all_b, all_sc, all_pref, all_allowed = [], [], [], []
        for b in range(B):
            for sc, pref in cands[b]:
                allowed = trie.get(pref)
                if not allowed:
                    continue
                all_b.append(b); all_sc.append(sc); all_pref.append(pref)
                all_allowed.append(sorted(allowed))
        if not all_b:
            break
        maxL = max(len(p) for p in all_pref) + 1
        tgt = torch.full((len(all_b), maxL), BOS, dtype=torch.long, device=device)
        for j, p in enumerate(all_pref):
            for t, v in enumerate(p):
                tgt[j, 1 + t] = v
        logits = model.decode(mem[all_b], tgt, pad[all_b])[:, -1, :]
        new = [[] for _ in range(B)]
        for j, (b, sc, pref) in enumerate(zip(all_b, all_sc, all_pref)):
            ids = torch.tensor(all_allowed[j], dtype=torch.long, device=device)
            vals = logits[j, ids]
            k = min(beams, ids.numel())
            tv, ti = vals.topk(k)
            for v, i in zip(tv.tolist(), ti.tolist()):
                new[b].append((sc + v, pref + (int(ids[i]),)))
        for b in range(B):
            new[b].sort(key=lambda x: -x[0])
            cands[b] = new[b][:beams]
    return cands


@torch.no_grad()
def evaluate(model, ds, trie, ks=(1, 3, 5, 10, 20, 50), beams=50, batch_size=32, device="cuda:0"):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    n = 0
    for src, tgt in loader:
        cands = beam_search(model, src, trie, beams=beams, device=device)
        for b in range(src.size(0)):
            target_ids = tuple(int(x) for x in tgt[b].tolist())
            ranked = [pref for _, pref in cands[b]]
            r = ranked.index(target_ids) if target_ids in ranked else None
            n += 1
            if r is None:
                continue
            for k in ks:
                if r < k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / math.log2(r + 2)
    return {f"HR@{k}": hits[k] / n for k in ks}, {f"NDCG@{k}": ndcg[k] / n for k in ks}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Sports_and_Outdoors")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beams", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--out_dir", default="experiments_tiger")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    base = os.path.join(args.data_dir, args.category, "sft")
    train_rows = load_samples(os.path.join(base, "train.json"))
    valid_rows = load_samples(os.path.join(base, "valid.json"))
    test_rows = load_samples(os.path.join(base, "test.json"))
    tokens, tok2id = build_vocab([train_rows, valid_rows, test_rows])
    id2tok = {v: k for k, v in tok2id.items()}
    vocab = len(tokens) + 3
    if rank == 0:
        print(f"train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} sid_tokens={len(tokens)} vocab={vocab}", flush=True)

    train_ds = SIDDataset(train_rows, tok2id)
    valid_ds = SIDDataset(valid_rows, tok2id)
    test_ds = SIDDataset(test_rows, tok2id)
    trie = build_trie(train_rows + valid_rows + test_rows, tok2id)

    model = Tiger(vocab, args.d_model, args.nhead, args.layers, args.layers,
                  args.dim_ff, args.dropout).to(device)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    core = model.module if ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, collate_fn=collate, num_workers=4, drop_last=False)
    vloader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate, num_workers=4)
    os.makedirs(args.out_dir, exist_ok=True)

    best = float("inf")
    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot, cnt = 0.0, 0
        t0 = time.time()
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt)
            loss = F.cross_entropy(out.reshape(-1, vocab), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * src.size(0); cnt += src.size(0)
        sched.step()
        if rank == 0:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for src, tgt in vloader:
                    src, tgt = src.to(device), tgt.to(device)
                    out = core(src, tgt)
                    vl += F.cross_entropy(out.reshape(-1, vocab), tgt.reshape(-1), ignore_index=PAD).item() * src.size(0)
                    vn += src.size(0)
            vl /= max(1, vn)
            print(f"[ep {ep+1}/{args.epochs}] train_loss={tot/max(1,cnt):.4f} valid_loss={vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if vl < best:
                best = vl
                torch.save(core.state_dict(), os.path.join(args.out_dir, f"tiger_{args.category}.pt"))
        if ddp:
            dist.barrier()

    if rank == 0:
        core.load_state_dict(torch.load(os.path.join(args.out_dir, f"tiger_{args.category}.pt"), map_location=device))
        hr, ndcg, n = evaluate(core, test_ds, trie, beams=args.beams, batch_size=args.eval_batch, device=device)
        res = {**hr, **ndcg, "n": n, "best_valid_loss": best, "sid_tokens": len(tokens)}
        print("RESULT", json.dumps(res), flush=True)
        with open(os.path.join(args.out_dir, f"metrics_tiger_{args.category}.json"), "w") as f:
            json.dump(res, f, indent=2)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
