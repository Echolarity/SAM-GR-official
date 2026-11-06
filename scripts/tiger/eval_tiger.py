import sys, os, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tiger import load_samples, build_vocab, SIDDataset, Tiger, build_trie, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--category", default="Sports_and_Outdoors")
ap.add_argument("--data_dir", default="./data")
ap.add_argument("--ckpt", default="experiments_tiger/tiger_Sports_and_Outdoors.pt")
ap.add_argument("--out", default="experiments_tiger/metrics_tiger_Sports_and_Outdoors.json")
ap.add_argument("--beams", type=int, default=50)
ap.add_argument("--eval_batch", type=int, default=32)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

base = os.path.join(args.data_dir, args.category, "sft")
train_rows = load_samples(os.path.join(base, "train.json"))
valid_rows = load_samples(os.path.join(base, "valid.json"))
test_rows = load_samples(os.path.join(base, "test.json"))
tokens, tok2id = build_vocab([train_rows, valid_rows, test_rows])
vocab = len(tokens) + 3
test_ds = SIDDataset(test_rows, tok2id)
trie = build_trie(train_rows + valid_rows + test_rows, tok2id)
model = Tiger(vocab).to(args.device)
model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
hr, ndcg, n = evaluate(model, test_ds, trie, beams=args.beams, batch_size=args.eval_batch, device=args.device)
res = {**hr, **ndcg, "n": n, "sid_tokens": len(tokens)}
print("RESULT", json.dumps(res), flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
