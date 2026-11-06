"""One downstream pipeline for an OMoE representation / intervention.

Stages: (optional RQ-VAE) -> SID index -> SFT data -> generator SFT -> eval.
The RQ-VAE can be reused from another run, which isolates the effect of an
input intervention from RQ-VAE retraining noise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.dirname(os.path.dirname(ROOT))
CATEGORY = "Sports_and_Outdoors"
DATA = os.path.join(BASE, "data", "Amazon", CATEGORY)
COLLAB = os.path.join(BASE, "sasrec_models", CATEGORY, "sasrec_item_emb.npy")
PY = "/data/sht/conda/mimigenrec/bin/python"
LLAMAFACTORY = "/data/sht/conda/mimigenrec/bin/llamafactory-cli"


def run(cmd, log_path, env=None, cwd=BASE):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    e = os.environ.copy()
    e.update({
        "PATH": "/data/sht/conda/mimigenrec/bin:" + e.get("PATH", ""),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline", "DISABLE_VERSION_CHECK": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    if env:
        e.update(env)
    print(f"[run] {cmd}\n  log={log_path}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n\n===== {time.strftime('%F %T')} {cmd} =====\n")
        f.flush()
        rc = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=f,
                              stderr=subprocess.STDOUT).wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd} (log {log_path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--rq-ckpt", default="")
    ap.add_argument("--rq-epochs", type=int, default=300)
    ap.add_argument("--merge-ratio", type=float, default=0.15)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--eval-gpus", default="0")
    ap.add_argument("--sft-steps", type=int, default=200)
    ap.add_argument("--sft-eval-steps", type=int, default=200)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--sft-grad-accum", type=int, default=16)
    ap.add_argument("--sft-lr", type=float, default=3.0e-4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-beams", type=int, default=20)
    ap.add_argument("--eval-max-samples", type=int, default=0,
                    help="0 = all; otherwise a deterministic systematic subset")
    ap.add_argument("--eval-split", default="test", choices=["test", "valid"])
    ap.add_argument("--stages", default="rq,index,preprocess,sft,eval,collect")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ws = os.path.join(ROOT, "results", "downstream", args.name)
    logs = os.path.join(ws, "logs")
    os.makedirs(logs, exist_ok=True)
    emb = os.path.join(ws, "emb.npy")
    if not os.path.exists(emb):
        os.symlink(os.path.abspath(args.emb), emb)
    rq_dir = os.path.join(ws, "rq")
    index_dir = os.path.join(ws, "index")
    data_out = os.path.join(ws, "data")
    sft_dir = os.path.join(ws, "sft_model")
    eval_dir = os.path.join(ws, "eval")
    for d in (rq_dir, index_dir, data_out, eval_dir):
        os.makedirs(d, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    timings = {}

    def timed(name, fn):
        t0 = time.time()
        fn()
        timings[name] = time.time() - t0
        print(f"[stage] {name} {timings[name]/60:.1f} min", flush=True)

    rq_ckpt = os.path.join(rq_dir, "best_collision_model.pth")
    index_path = os.path.join(index_dir, f"{CATEGORY}.index.json")

    if "rq" in stages and not args.rq_ckpt:
        if args.force and os.path.exists(rq_ckpt):
            os.remove(rq_ckpt)
        if not os.path.exists(rq_ckpt):
            def _rq():
                run([PY, os.path.join(BASE, "rq", "rqvae.py"),
                     "--data_path", emb, "--ckpt_dir", rq_dir, "--lr", "1e-3",
                     "--epochs", str(args.rq_epochs), "--batch_size", "20480",
                     "--num_emb_list", "256", "256", "256", "256",
                     "--sk_epsilons", "0.0", "0.0", "0.0", "0.0",
                     "--device", "cuda:0"],
                    os.path.join(logs, "rqvae.log"),
                    env={"CUDA_VISIBLE_DEVICES": args.gpus.split(",")[0]})
            timed("rq", _rq)
    if args.rq_ckpt:
        if os.path.abspath(args.rq_ckpt) != os.path.abspath(rq_ckpt):
            shutil.copy(args.rq_ckpt, rq_ckpt)
        import torch
        ck = torch.load(rq_ckpt, map_location="cpu", weights_only=False)
        ck["args"].data_path = os.path.abspath(emb)
        torch.save(ck, rq_ckpt)

    if "index" in stages:
        if args.force or not os.path.exists(index_path):
            def _idx():
                run([PY, os.path.join(HERE, "make_sid.py"),
                     "--data_path", emb, "--ckpt", rq_ckpt,
                     "--out_dir", index_dir, "--name", CATEGORY,
                     "--device", "cuda:0", "--merge_ratio", str(args.merge_ratio),
                     "--collab_emb_path", COLLAB,
                     "--train_inter", os.path.join(DATA, f"{CATEGORY}.train.inter")],
                    os.path.join(logs, "index.log"),
                    env={"CUDA_VISIBLE_DEVICES": args.gpus.split(",")[0]})
            timed("index", _idx)

    if "preprocess" in stages:
        marker = os.path.join(data_out, "sft", "train.json")
        if args.force or not os.path.exists(marker):
            def _pre():
                amazon = os.path.join(ws, "amazon", CATEGORY)
                os.makedirs(amazon, exist_ok=True)
                for fn in [f"{CATEGORY}.item.json", f"{CATEGORY}.train.inter",
                           f"{CATEGORY}.valid.inter", f"{CATEGORY}.test.inter"]:
                    dst = os.path.join(amazon, fn)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(os.path.join(DATA, fn), dst)
                dst_idx = os.path.join(amazon, f"{CATEGORY}.index.json")
                if os.path.lexists(dst_idx):
                    os.remove(dst_idx)
                shutil.copy(index_path, dst_idx)
                run([PY, "-m", "samgr.preprocess",
                     "--data_dir", os.path.join(ws, "amazon"),
                     "--category", CATEGORY, "--output_dir", data_out,
                     "--only_task1", "True", "--sid_num", "4", "--seed", "42"],
                    os.path.join(logs, "preprocess.log"))
            timed("preprocess", _pre)

    if "sft" in stages:
        done = os.path.join(sft_dir, "config.json")
        if args.force or not os.path.exists(done):
            def _sft():
                ds = f"lb_{args.name.replace('-', '_')}"
                info = {
                    f"{ds}_train": {"file_name": "sft/train.json",
                                    "columns": {"prompt": "instruction", "query": "input",
                                                "response": "output", "system": "system"}},
                    f"{ds}_valid": {"file_name": "sft/valid.json",
                                    "columns": {"prompt": "instruction", "query": "input",
                                                "response": "output", "system": "system"}},
                }
                json.dump(info, open(os.path.join(data_out, "dataset_info.json"), "w"), indent=2)
                yml = os.path.join(ws, "sft.yaml")
                text = f"""### model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
add_tokens_list: '{os.path.join(data_out, "new_tokens.json")}'
### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json
### dataset
dataset: {ds}_train
eval_dataset: {ds}_valid
dataset_dir: {data_out}
cutoff_len: 512
preprocessing_num_workers: 8
dataloader_num_workers: 4
### output
output_dir: {sft_dir}
logging_steps: 1
save_steps: {args.sft_eval_steps}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
save_strategy: steps
load_best_model_at_end: false
save_total_limit: 1
### experiment
report_to: none
run_name: lb-{args.name}
### train
per_device_train_batch_size: {args.sft_batch}
gradient_accumulation_steps: {args.sft_grad_accum}
learning_rate: {args.sft_lr}
max_steps: {args.sft_steps}
lr_scheduler_type: cosine
warmup_steps: 10
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
metric_for_best_model: loss
### eval
per_device_eval_batch_size: 8
eval_strategy: "no"
eval_steps: {args.sft_eval_steps}
"""
                open(yml, "w").write(text)
                run([LLAMAFACTORY, "train", yml], os.path.join(logs, "sft.log"),
                    env={"CUDA_VISIBLE_DEVICES": args.gpus, "FORCE_TORCHRUN": "1"})
            timed("sft", _sft)

    if "eval" in stages:
        metrics_path = os.path.join(eval_dir, f"metrics_{args.eval_split}.json")
        if args.force or not os.path.exists(metrics_path):
            def _eval():
                temp = os.path.join(ws, f"eval_tmp_{args.eval_split}")
                shutil.rmtree(temp, ignore_errors=True)
                os.makedirs(temp, exist_ok=True)
                src = os.path.join(data_out, "sft", f"{args.eval_split}.json")
                gpus = [g for g in args.eval_gpus.split(",") if g]
                data = json.load(open(src))
                if args.eval_max_samples and len(data) > args.eval_max_samples:
                    step = max(1, len(data) // args.eval_max_samples)
                    data = data[::step][:args.eval_max_samples]
                parts = {}
                for k, g in enumerate(gpus):
                    start = k * len(data) // len(gpus)
                    end = (k + 1) * len(data) // len(gpus)
                    parts[g] = data[start:end]
                    json.dump(parts[g], open(os.path.join(temp, f"{g}.json"), "w"),
                              ensure_ascii=False)
                procs = []
                for k, g in enumerate(gpus):
                    part = os.path.join(temp, f"{g}.json")
                    cmd = [PY, "-u", "-m", "samgr.evaluate",
                           "--model_name_or_path", sft_dir,
                           "--test_data_path", part,
                           "--result_json_path", os.path.join(temp, f"{g}_result.json"),
                           "--index_path", index_path,
                           "--batch_size", str(args.eval_batch),
                           "--num_beams", str(args.eval_beams),
                           "--max_new_tokens", "64", "--temperature", "1.0",
                           "--do_sample", "False", "--length_penalty", "0.0",
                           "--seed", str(42 + k), "--compute_metrics_flag", "False",
                           "--add_tokens_path", os.path.join(data_out, "new_tokens.json")]
                    e = os.environ.copy()
                    e.update({"PATH": "/data/sht/conda/mimigenrec/bin:" + e.get("PATH", ""),
                              "CUDA_VISIBLE_DEVICES": g, "HF_HUB_OFFLINE": "1",
                              "TRANSFORMERS_OFFLINE": "1",
                              "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
                    lf = open(os.path.join(logs, f"eval_{args.eval_split}_gpu{g}.log"), "a")
                    procs.append((subprocess.Popen(cmd, cwd=BASE, env=e, stdout=lf,
                                                   stderr=subprocess.STDOUT), lf))
                for p, lf in procs:
                    rc = p.wait()
                    lf.close()
                    if rc != 0:
                        raise RuntimeError("eval failed")
                merged = os.path.join(eval_dir, f"final_result_{args.eval_split}.json")
                out = []
                for g in gpus:
                    p = os.path.join(temp, f"{g}_result.json")
                    if os.path.exists(p):
                        out.extend(json.load(open(p)))
                json.dump(out, open(merged, "w"), ensure_ascii=False)
                run([PY, "-m", "samgr.evaluate",
                     "--result_json_path", merged, "--metrics_only", "True"],
                    os.path.join(logs, f"eval_metrics_{args.eval_split}.log"))
                src_m = os.path.join(eval_dir, "metrics.json")
                if os.path.exists(src_m):
                    shutil.copy(src_m, metrics_path)
                shutil.rmtree(temp, ignore_errors=True)
            timed("eval", _eval)

    if "collect" in stages:
        rq_meta = json.load(open(os.path.join(index_dir, f"{CATEGORY}.metrics.json"))) \
            if os.path.exists(os.path.join(index_dir, f"{CATEGORY}.metrics.json")) else {}
        ev = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        tr = {}
        if os.path.exists(os.path.join(sft_dir, "train_results.json")):
            tr = json.load(open(os.path.join(sft_dir, "train_results.json")))
        row = {
            "name": args.name, "emb": os.path.abspath(args.emb),
            "rq_ckpt": args.rq_ckpt or rq_ckpt, "rq_epochs": args.rq_epochs,
            "sft_steps": args.sft_steps, "eval_split": args.eval_split,
            "eval_beams": args.eval_beams,
            "HR@5": ev.get("HR@5"), "HR@10": ev.get("HR@10"), "HR@20": ev.get("HR@20"),
            "NDCG@5": ev.get("NDCG@5"), "NDCG@10": ev.get("NDCG@10"),
            "NDCG@20": ev.get("NDCG@20"),
            "unique_sids": rq_meta.get("unique_sids"),
            "excess_collision_rate": rq_meta.get("excess_collision_rate"),
            "collided_item_ratio": rq_meta.get("collided_item_ratio"),
            "max_bucket": rq_meta.get("max_bucket"),
            "raw_collision_rate": rq_meta.get("raw_collision_rate"),
            "codebook_perplexity_mean": rq_meta.get("codebook_perplexity_mean"),
            "recon_mse": rq_meta.get("recon_mse"),
            "train_runtime": tr.get("train_runtime"),
            "timings": timings,
        }
        os.makedirs(os.path.join(ROOT, "results", "downstream"), exist_ok=True)
        out = os.path.join(ROOT, "results", "downstream", f"{args.name}.json")
        json.dump(row, open(out, "w"), indent=2)
        print(json.dumps(row, indent=2), flush=True)
    print(f"DOWNSTREAM_DONE {args.name}", flush=True)


if __name__ == "__main__":
    main()
