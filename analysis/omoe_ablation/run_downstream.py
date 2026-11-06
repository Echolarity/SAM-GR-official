"""Downstream retraining for one OMoE variant: RQ-VAE -> SID -> generator -> eval."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(HERE, "workspaces")
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(BASE, "data", "Amazon", "Sports_and_Outdoors")
COLLAB = os.path.join(BASE, "sasrec_models", "Sports_and_Outdoors", "sasrec_item_emb.npy")
CATEGORY = "Sports_and_Outdoors"
PY = os.path.join("/data/sht/conda/mimigenrec/bin", "python")
LLAMAFACTORY = os.path.join("/data/sht/conda/mimigenrec/bin", "llamafactory-cli")


def run(cmd: list[str], log_path: str, env: dict | None = None, cwd: str = BASE) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    e = os.environ.copy()
    e.update({
        "PATH": "/data/sht/conda/mimigenrec/bin:" + e.get("PATH", ""),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "DISABLE_VERSION_CHECK": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    if env:
        e.update(env)
    print(f"[run] {cmd}\n  log={log_path}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"\n\n===== {time.strftime('%F %T')} {cmd} =====\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd} (log {log_path})")


def stage_done(path: str, skip: bool) -> bool:
    return skip and os.path.exists(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--rq-epochs", type=int, default=2000)
    ap.add_argument("--sft-max-steps", type=int, default=60)
    ap.add_argument("--sft-epochs", type=float, default=0.0)
    ap.add_argument("--sft-eval-steps", type=int, default=100)
    ap.add_argument("--sft-save-steps", type=int, default=100)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--sft-grad-accum", type=int, default=16)
    ap.add_argument("--sft-lr", type=float, default=3.0e-4)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--eval-gpus", default="0,1,2,3")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-beams", type=int, default=20)
    ap.add_argument("--stages", default="rq,index,preprocess,sft,eval,collect")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    v = args.variant
    ws = os.path.join(WORK, v)
    emb = os.path.join(ws, "emb.npy")
    if not os.path.exists(emb):
        raise SystemExit(f"missing embedding: {emb}")
    rq_dir = os.path.join(ws, "rq")
    index_dir = os.path.join(ws, "index")
    index_path = os.path.join(index_dir, f"{CATEGORY}.index.json")
    data_out = os.path.join(ws, "data")
    amazon_cat = os.path.join(ws, "amazon", CATEGORY)
    sft_dir = os.path.join(ws, "sft_model")
    eval_dir = os.path.join(ws, "eval")
    log_dir = os.path.join(ws, "logs")
    timings_path = os.path.join(ws, "timings.json")
    os.makedirs(rq_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(data_out, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timings = json.load(open(timings_path)) if os.path.exists(timings_path) else {}
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    def timed(name, fn):
        t0 = time.time()
        fn()
        timings[name] = timings.get(name, 0.0) + (time.time() - t0)
        json.dump(timings, open(timings_path, "w"), indent=2)
        print(f"[stage] {name} done in {timings[name]/60:.1f} min", flush=True)

    # ---------------- RQ-VAE ----------------
    if "rq" in stages:
        ckpt = os.path.join(rq_dir, "best_collision_model.pth")
        if args.force and os.path.exists(ckpt):
            os.remove(ckpt)
        if not (args.skip_existing and os.path.exists(ckpt)):
            def _rq():
                run([
                    PY, os.path.join(BASE, "rq", "rqvae.py"),
                    "--data_path", emb,
                    "--ckpt_dir", rq_dir,
                    "--lr", "1e-3",
                    "--epochs", str(args.rq_epochs),
                    "--batch_size", "20480",
                    "--num_emb_list", "256", "256", "256", "256",
                    "--sk_epsilons", "0.0", "0.0", "0.0", "0.0",
                    "--device", "cuda:0",
                ], os.path.join(log_dir, "rqvae.log"), env={"CUDA_VISIBLE_DEVICES": args.gpus.split(",")[0]})
            timed("rq", _rq)

    # ---------------- SID indices ----------------
    if "index" in stages:
        if not (args.skip_existing and os.path.exists(index_path)):
            def _idx():
                run([
                    PY, os.path.join(BASE, "rq", "generate_indices.py"),
                    "--dataset", CATEGORY,
                    "--ckpt_path", os.path.join(rq_dir, "best_collision_model.pth"),
                    "--output_dir", index_dir,
                    "--device", "cuda:0",
                    "--collab_emb_path", COLLAB,
                    "--train_inter", os.path.join(DATA, f"{CATEGORY}.train.inter"),
                    "--merge_ratio", "0.15",
                ], os.path.join(log_dir, "index.log"), env={"CUDA_VISIBLE_DEVICES": args.gpus.split(",")[0]})
            timed("index", _idx)

    # ---------------- preprocess ----------------
    if "preprocess" in stages:
        if not (args.skip_existing and os.path.exists(os.path.join(data_out, "sft", "train.json"))):
            def _pre():
                os.makedirs(amazon_cat, exist_ok=True)
                for fn in [f"{CATEGORY}.item.json", f"{CATEGORY}.train.inter",
                           f"{CATEGORY}.valid.inter", f"{CATEGORY}.test.inter"]:
                    dst = os.path.join(amazon_cat, fn)
                    src = os.path.join(DATA, fn)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
                dst_idx = os.path.join(amazon_cat, f"{CATEGORY}.index.json")
                if os.path.lexists(dst_idx):
                    os.remove(dst_idx)
                shutil.copy(index_path, dst_idx)
                run([
                    PY, "-m", "samgr.preprocess",
                    "--data_dir", os.path.join(ws, "amazon"),
                    "--category", CATEGORY,
                    "--output_dir", data_out,
                    "--only_task1", "True",
                    "--sid_num", "4",
                    "--seed", "42",
                ], os.path.join(log_dir, "preprocess.log"))
            timed("preprocess", _pre)

    # ---------------- SFT ----------------
    if "sft" in stages:
        done_marker = os.path.join(sft_dir, "config.json")
        if not (args.skip_existing and os.path.exists(done_marker)):
            def _sft():
                ds = f"abl_{v}"
                info = {
                    f"{ds}_train": {
                        "file_name": "sft/train.json",
                        "columns": {"prompt": "instruction", "query": "input",
                                    "response": "output", "system": "system"},
                    },
                    f"{ds}_valid": {
                        "file_name": "sft/valid.json",
                        "columns": {"prompt": "instruction", "query": "input",
                                    "response": "output", "system": "system"},
                    },
                }
                json.dump(info, open(os.path.join(data_out, "dataset_info.json"), "w"), indent=2)
                yml = os.path.join(ws, "sft.yaml")
                steps_line = (f"max_steps: {args.sft_max_steps}\n"
                              if args.sft_max_steps and args.sft_max_steps > 0
                              else f"num_train_epochs: {args.sft_epochs}\n")
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
preprocessing_num_workers: 16
dataloader_num_workers: 4
### output
output_dir: {sft_dir}
logging_steps: 1
save_steps: {args.sft_save_steps}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
save_strategy: steps
load_best_model_at_end: true
save_total_limit: 1
### experiment
report_to: none
run_name: abl-{v}
### train
per_device_train_batch_size: {args.sft_batch}
gradient_accumulation_steps: {args.sft_grad_accum}
learning_rate: {args.sft_lr}
{steps_line}lr_scheduler_type: cosine
warmup_steps: 10
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
metric_for_best_model: loss
### eval
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: {args.sft_eval_steps}
"""
                open(yml, "w").write(text)
                run([LLAMAFACTORY, "train", yml], os.path.join(log_dir, "sft.log"),
                    env={"CUDA_VISIBLE_DEVICES": args.gpus})
            timed("sft", _sft)

    # ---------------- eval ----------------
    if "eval" in stages:
        metrics_path = os.path.join(eval_dir, "metrics.json")
        if not (args.skip_existing and os.path.exists(metrics_path)):
            def _eval():
                temp = os.path.join(ws, "eval_tmp")
                shutil.rmtree(temp, ignore_errors=True)
                os.makedirs(temp, exist_ok=True)
                test_json = os.path.join(data_out, "sft", "test.json")
                cuda_list = args.eval_gpus.replace(",", " ")
                run([PY, "-m", "samgr.split_json",
                     "--input_path", test_json, "--output_path", temp,
                     "--cuda_list", cuda_list], os.path.join(log_dir, "eval_split.log"))
                procs = []
                for i, g in enumerate(args.eval_gpus.split(",")):
                    part = os.path.join(temp, f"{g}.json")
                    if not os.path.exists(part):
                        continue
                    cmd = [
                        PY, "-u", "-m", "samgr.evaluate",
                        "--model_name_or_path", sft_dir,
                        "--test_data_path", part,
                        "--result_json_path", os.path.join(temp, f"{g}_result.json"),
                        "--index_path", index_path,
                        "--batch_size", str(args.eval_batch),
                        "--num_beams", str(args.eval_beams),
                        "--max_new_tokens", "64",
                        "--temperature", "1.0",
                        "--do_sample", "False",
                        "--length_penalty", "0.0",
                        "--seed", str(42 + i),
                        "--compute_metrics_flag", "False",
                        "--add_tokens_path", os.path.join(data_out, "new_tokens.json"),
                    ]
                    e = os.environ.copy()
                    e.update({
                        "PATH": "/data/sht/conda/mimigenrec/bin:" + e.get("PATH", ""),
                        "CUDA_VISIBLE_DEVICES": g,
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    })
                    lf = open(os.path.join(log_dir, f"eval_gpu{g}.log"), "a")
                    lf.write(f"\n===== {time.strftime('%F %T')} eval gpu {g} =====\n")
                    procs.append((subprocess.Popen(cmd, cwd=BASE, env=e, stdout=lf, stderr=subprocess.STDOUT), lf))
                for p, lf in procs:
                    rc = p.wait()
                    lf.close()
                    if rc != 0:
                        raise RuntimeError(f"eval failed rc={rc}")
                merged = os.path.join(eval_dir, "final_result.json")
                run([PY, "-m", "samgr.merge_json",
                     "--input_path", temp, "--output_path", merged,
                     "--cuda_list", cuda_list, "--compute_metrics", "False"],
                    os.path.join(log_dir, "eval_merge.log"))
                run([PY, "-m", "samgr.evaluate",
                     "--result_json_path", merged, "--metrics_only", "True"],
                    os.path.join(log_dir, "eval_metrics.log"))
                src_metrics = os.path.join(eval_dir, "metrics.json")
                if os.path.exists(src_metrics) and os.path.abspath(src_metrics) != os.path.abspath(metrics_path):
                    shutil.copy(src_metrics, metrics_path)
                shutil.rmtree(temp, ignore_errors=True)
            timed("eval", _eval)

    # ---------------- collect ----------------
    if "collect" in stages:
        rep_path = os.path.join(RESULTS, "representation", f"{v}.json")
        rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
        rq_path = os.path.join(ws, "rq_metrics.json")
        if os.path.exists(os.path.join(rq_dir, "best_collision_model.pth")) and (
            args.force or not os.path.exists(rq_path)
        ):
            run([PY, os.path.join(HERE, "rq_metrics.py"),
                 "--data_path", emb,
                 "--ckpt", os.path.join(rq_dir, "best_collision_model.pth"),
                 "--index_path", index_path,
                 "--out", rq_path,
                 "--device", "cuda:0"],
                os.path.join(log_dir, "rq_metrics.log"),
                env={"CUDA_VISIBLE_DEVICES": args.gpus.split(",")[0]})
        rq = json.load(open(rq_path)) if os.path.exists(rq_path) else {}
        ev = json.load(open(os.path.join(eval_dir, "metrics.json"))) if os.path.exists(
            os.path.join(eval_dir, "metrics.json")) else {}
        sft_runtime = None
        tr_path = os.path.join(sft_dir, "train_results.json")
        if os.path.exists(tr_path):
            try:
                sft_runtime = json.load(open(tr_path)).get("train_runtime")
            except Exception:
                pass
        row = {
            "variant": v,
            "group": rep.get("group", ""),
            "notes": rep.get("notes", ""),
            "HR@5": ev.get("HR@5"), "NDCG@5": ev.get("NDCG@5"),
            "HR@10": ev.get("HR@10"), "NDCG@10": ev.get("NDCG@10"),
            "HR@20": ev.get("HR@20"), "NDCG@20": ev.get("NDCG@20"),
            "HR@1": ev.get("HR@1"), "NDCG@1": ev.get("NDCG@1"),
            "HR@50": ev.get("HR@50"), "NDCG@50": ev.get("NDCG@50"),
            "rq_params": rq.get("rq_params"),
            "recon_mse": rq.get("recon_mse"),
            "recon_rmse": rq.get("recon_rmse"),
            "codebook_perplexity_mean": rq.get("mean_codebook_perplexity"),
            "codebook_perplexity_min": rq.get("min_codebook_perplexity"),
            "codebook_perplexity_max": rq.get("max_codebook_perplexity"),
            "raw_collision_rate": rq.get("raw_collision_rate"),
            "final_collision_rate": rq.get("final_collision_rate"),
            "final_unique_sids": rq.get("final_unique_sids"),
            "expert_mean_abs_cos": rep.get("mean_abs_expert_cos"),
            "expert_max_abs_cos": rep.get("max_abs_expert_cos"),
            "gate_load_cv": rep.get("gate_load_cv"),
            "batch_load_cv": rep.get("batch_load_cv"),
            "gate_entropy": rep.get("gate_entropy"),
            "item_routing_entropy": rep.get("mean_item_entropy"),
            "item_routing_entropy_std": rep.get("std_item_entropy"),
            "routing_item_var": rep.get("routing_item_var"),
            "routing_total_var": rep.get("routing_total_var"),
            "fused_std": rep.get("fused_std"),
            "fused_mean_cos": rep.get("fused_mean_cos"),
            "fused_nn_dup": rep.get("fused_nn_dup"),
            "direct_share": rep.get("direct_share"),
            "moe_share": rep.get("moe_share"),
            "cos_direct_moe": rep.get("cos_direct_moe"),
            "rel_diff_no_moe": rep.get("rel_diff_no_moe"),
            "rel_diff_no_direct": rep.get("rel_diff_no_direct"),
            "aligned_norm_share": rep.get("aligned_norm_share"),
            "branch_norm_share": rep.get("branch_norm_share"),
            "cos_aligned_branch": rep.get("cos_aligned_branch"),
            "params": rep.get("params"),
            "params_M": (rep.get("params") / 1e6) if rep.get("params") else None,
            "omoe_train_time_sec": rep.get("train_time_sec"),
            "rq_train_time_sec": timings.get("rq"),
            "sft_train_time_sec": sft_runtime if sft_runtime is not None else timings.get("sft"),
            "eval_time_sec": timings.get("eval"),
            "total_time_sec": sum(timings.values()),
            "sft_max_steps": args.sft_max_steps,
            "rq_epochs": rq.get("rq_epochs", args.rq_epochs),
            "eval_beams": args.eval_beams,
        }
        out = os.path.join(RESULTS, f"downstream_{v}.json")
        json.dump(row, open(out, "w"), indent=2)
        print(json.dumps(row, indent=2), flush=True)

    print(f"DOWNSTREAM_DONE {v}", flush=True)


if __name__ == "__main__":
    main()
