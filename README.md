<p align="center">
  <img src="assets/MIMIGenRec_logo.png" alt="SAM-GR logo"/>
</p>

<p align="center"><strong>SAM-GR: Multimodal Generative Recommendation with Collaborative-Semantic Aggregation and Mixture-of-Experts Fusion</strong></p>

<p align="center"><strong>Built on <a href="https://github.com/cocoshe/MIMIGenRec">MIMIGenRec</a></strong></p>

---

## Table of contents

1. [Overview](#1-overview)
2. [What is inherited vs. new](#2-what-is-inherited-vs-new)
3. [Repository layout](#3-repository-layout)
4. [Requirements](#4-requirements)
5. [Installation](#5-installation)
6. [Smoke test](#6-smoke-test)
7. [Data preparation](#7-data-preparation)
8. [Stage-by-stage reproduction](#8-stage-by-stage-reproduction)
9. [One-click end-to-end reproduction](#9-one-click-end-to-end-reproduction)
10. [SFT](#10-sft)
11. [RL with GRPO](#11-rl-with-grpo)
12. [Evaluation](#12-evaluation)
13. [Baselines](#13-baselines)
14. [Ablations and analysis](#14-ablations-and-analysis)
15. [Generated artifacts](#15-generated-artifacts)
16. [Troubleshooting](#16-troubleshooting)
17. [Reproducibility checklist](#17-reproducibility-checklist)
18. [License and acknowledgments](#18-license-and-acknowledgments)

---

## 1. Overview

SAM-GR is a generative recommendation pipeline that turns raw Amazon reviews,
metadata, and product images into a fine-tuned (and optionally RL-tuned)
item-ID generator. It combines:

- **Multimodal item representations** from Qwen3-VL (text + image).
- **Collaborative-semantic alignment (CSGA)** using SASRec item-item graphs.
- **Mixture-of-Experts fusion** (OMoE / OMoE-v2) of text, image and collaborative features.
- **RQ-VAE semantic IDs (SIDs)** with 4 hierarchical tokens per item.
- **LLaMA-Factory SFT** and **TRL GRPO** training over constrained SID generation.
- **Trie-constrained beam-search evaluation** with HR / NDCG metrics.

End-to-end data flow:

```
raw reviews + metadata + images (data/)
        │
  ① possess_data ──────────────► item.json + *.inter + image paths
        │
  ② SASRec (collab modes only) ► sasrec_item_emb.npy
        │
  ③ multimodal text2emb ───────► {Category}.emb-<model>-<mode>.npy
        │    (Qwen3-VL + optional CSGA / OMoE fusion)
  ④ RQ-VAE (4-level codebook) ─► best_collision_model.pth
        │
  ⑤ generate_indices ──────────► {Category}.index.json
        │
  ⑥ preprocess (sid_num=4) ────► sft/ + rl/ + new_tokens.json + id2sid.json
        │
  ⑦ SFT (LLaMA-Factory) ───────► newsaves/...
        │
  (optional) RL (TRL GRPO) ────► rl_outputs/...
        │
  evaluation ──────────────────► results/.../metrics.json
```

---

## 2. What is inherited vs. new

### Inherited from MIMIGenRec

- LLaMA-Factory-based SFT (Qwen2.5 0.5B / 1.5B / 3B, full / LoRA, DeepSpeed ZeRO).
- TRL `GRPOTrainer` with ranking rewards (NDCG / HR) and constrained beam search.
- RQ-VAE semantic IDs and the `new_tokens.json` / `id2sid.json` formats.
- SFT/RL dataset schemas and dataset registry.

### New in this repository

- **Multimodal item embeddings** from Qwen3-VL using product text and images.
- **CSGA**: collaborative similarity graphs from SASRec guide semantic feature aggregation.
- **OMoE**: orthogonal Mixture-of-Experts with modality-specific and shared experts.
- **OMoE-v2**: discriminability-aware variant with near-duplicate repulsion and variance matching.
- **CSGA → OMoE pipelines** and **14 fusion modes** in one embedding script.
- **4-layer RQ-VAE SIDs** plus low-frequency SID merging (`SID_MERGE_RATIO`).
- **One-click 7-stage pipeline** with resume support and background execution.
- Additional datasets and updated baselines (GRU4Rec, Caser, SASRec).

---

## 3. Repository layout

```
SAM-GR/
├── samgr/                          # Core Python library and CLI entry points
│   ├── model.py                    # MIMIGenRec model wrapper
│   ├── logits.py                   # Constrained logits processor
│   ├── utils.py                    # Trie / tokenizer / config helpers
│   ├── evaluate.py                 # Trie-constrained beam-search evaluation
│   ├── preprocess.py               # SFT/RL dataset construction
│   ├── train_rl.py                 # TRL GRPOTrainer entry point
│   ├── merge_json.py               # Multi-GPU result merging
│   ├── split_json.py               # Multi-GPU test splitting
│   ├── inject_priors.py            # Codebook-prior injection
│   ├── check_seed.py               # Determinism check for evaluation
│   └── load_qwen_vl.py             # Qwen-VL checker / sanity inference
├── workflows/                      # Shell orchestration
│   ├── main.sh                     # 7-stage end-to-end pipeline
│   ├── nohup_run.sh                # Background job runner
│   ├── run.sh                      # start/stop/status wrapper for main.sh
│   ├── run_all.sh                  # env → data → SFT → RL → eval
│   ├── run_config.sh               # Central experiment configuration
│   ├── sft.sh                      # LlamaFactory SFT launcher
│   ├── trl_trainer.sh              # RL/GRPO launcher
│   ├── evaluate.sh                 # Evaluation launcher
│   ├── preprocess_data_sft_rl.sh   # Standalone preprocessing wrapper
│   ├── data/                       # Dataset preparation drivers
│   └── experiments/                # Per-dataset experiment drivers
├── src/llamafactory/               # LLaMA-Factory framework (upstream)
├── rq/                             # RQ-VAE, SID, text2emb, CSGA, OMoE
├── scripts/                        # Baselines (rec_zoo), TIGER, diagnostics
├── rewards/                        # GRPO ranking rewards
├── data/                           # Data scripts + datasets (datasets gitignored)
├── examples/                       # SFT YAMLs + DeepSpeed configs
├── config/                         # DeepSpeed configs used by RL
├── configs/                        # Generated local configs
├── analysis/                       # Analysis scripts, reports, ablation artifacts
├── experiments*/                   # Baseline experiment configs / checkpoints
├── tests/                          # Smoke test and stubs
├── assets/, docs/                  # Logo and documentation
├── pyproject.toml, Makefile, ...   # Packaging / lint
└── runs/, saves/, newsaves/, rq_output/, rl_outputs/, results/,
    sasrec_models/, wandb/, logs/, nohup_logs/, temp/   # Generated (gitignored)
```

---

## 4. Requirements

### Hardware

| Component | Recommended |
|-----------|-------------|
| GPUs | 4 × CUDA GPUs (24 GB+ each for 0.5B; 40 GB+ for 1.5B; 80 GB for 3B) |
| CPU RAM | 64 GB+ |
| Disk | Raw data + images: tens of GB; checkpoints and embeddings: 100 GB+ |

A single GPU works for small experiments, but embedding, RQ-VAE and SFT are much slower.

### Software

Tested versions:

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.8.0+cu128 |
| transformers | 4.57.1 |
| trl | 0.24.0 |
| accelerate | 1.11.0 |
| peft | 0.17.1 |
| datasets | 4.0.0 |

### Model checkpoints

- **VLM for item embeddings**: Qwen3-VL-8B-Instruct (or compatible Qwen3-VL checkpoint).
- **Generator for SFT/RL**: Qwen2.5-0.5B-Instruct / 1.5B / 3B.

Set paths explicitly; nothing is downloaded automatically except standard Hugging Face models.

---

## 5. Installation

### 5.1 Clone and enter the project

```bash
git clone git@github.com:Echolarity/SAM-GR.git
cd SAM-GR
```

### 5.2 Create a Python environment

```bash
conda create -n samgr python=3.12 -y
conda activate samgr
```

If you already have a working environment (for example `mimigenrec`), you can reuse it.

### 5.3 Install LLaMA-Factory in editable mode

```bash
pip install -e .
```

This provides the `llamafactory-cli` command and installs PyTorch, Transformers,
TRL, Accelerate, PEFT, Datasets, and the remaining dependencies.

### 5.4 Optional environment setup

```bash
# Hugging Face mirror (optional)
export HF_ENDPOINT=https://hf-mirror.com

# Weights & Biases (optional)
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=SAM-GR
export WANDB_MODE=offline   # or online
```

### 5.5 Verify the installation

```bash
python -c "import torch, transformers, trl, accelerate, peft, datasets; print('deps OK')"
python -c "import samgr.evaluate, samgr.preprocess, samgr.train_rl; print('samgr OK')"
```

---

## 6. Smoke test

Before running the full pipeline, verify the workflow wiring with the synthetic
smoke test:

```bash
bash tests/smoke_test.sh
```

Expected final line:

```
SMOKE TEST PASSED
```

What the smoke test does:

1. Generates a tiny synthetic Amazon-style dataset (48 reviews, 12 items, images).
2. Runs the **real** `data/possess_data.py` stage.
3. Runs `workflows/main.sh` with the full 7-stage control flow.
4. Runs the **real** SFT/RL preprocessing (`samgr.preprocess`) on the synthetic SIDs.
5. Runs the **real** `samgr.split_json` and `samgr.merge_json` multi-GPU plumbing.
6. Stubs only the GPU/model-heavy commands:
   - `torchrun` multimodal embedding,
   - RQ-VAE training,
   - SID index generation,
   - LlamaFactory SFT,
   - evaluation inference.

The smoke test takes seconds, needs no GPU, and leaves artifacts under
`temp/smoke_test/` (gitignored). Use it after any refactor to confirm that paths,
imports, environment variables, and stage contracts are intact.

To keep or inspect the artifacts:

```bash
find temp/smoke_test -maxdepth 3 -type f | sort
```

---

## 7. Data preparation

### 7.1 Raw data layout

Put raw files under `data/`:

```
data/
├── {Category}.jsonl.gz          # reviews
├── meta_{Category}.jsonl.gz     # item metadata
├── {Category}_images/           # one image per item: {asin}_MAIN.jpg
└── ...
```

Supported examples include:

- `Industrial_and_Scientific`
- `Sports_and_Outdoors`
- `Toys_and_Games`
- `Beauty_and_Personal_Care`
- `Books`
- `Office_Products`

Review records use the modern Amazon format:

```json
{"parent_asin": "...", "user_id": "...", "timestamp": 1700000000000,
 "rating": 5.0, "text": "...", "title": "..."}
```

Metadata records:

```json
{"parent_asin": "...", "title": "...", "description": "...",
 "store": "...", "categories": [["..."]],
 "images": [{"variant": "MAIN", "hi_res": "..."}]}
```

### 7.2 Dataset helper scripts

Available under `workflows/data/`:

| Script | Purpose |
|--------|---------|
| `download_cats.sh` | Integrity check, 5-core filtering, image list + high-concurrency download |
| `run_make_10core.sh` | Build strict 10-core Sports / Toys datasets |
| `run_make_10core_beauty.sh` | Build Beauty 10-core dataset |
| `run_make_strict.sh` | Build strict k-core datasets for Sports / Toys / Beauty |
| `run_10core_images.sh` | Ensure every retained item has one MAIN image |
| `run_kc_multi.sh` | Multi-dataset k-core statistics |

Example:

```bash
bash workflows/data/download_cats.sh
# or in the background
bash workflows/nohup_run.sh run cats_images_dl bash workflows/data/download_cats.sh
```

### 7.3 Stage 1 — build item/interaction files

```bash
python data/possess_data.py \
  --category Industrial_and_Scientific \
  --sample_size -1 \
  --user_k 5 --item_k 5 \
  --output_path ./data/Amazon \
  --raw_dir ./data \
  --img_dir ./data/Industrial_and_Scientific_images
```

Outputs under `data/Amazon/{Category}/`:

```
{Category}.item.json       # item_id -> {title, description, brand, categories, image_paths}
{Category}.train.inter
{Category}.valid.inter
{Category}.test.inter
{Category}.review.json
{Category}.user2id
{Category}.item2id
```

`item_id` values are contiguous integers `0..N-1`; downstream stages rely on this.

---

## 8. Stage-by-stage reproduction

Set common variables first:

```bash
export CATEGORY=Industrial_and_Scientific
export MODAL_CHOICE=csga_moe_collab_v2
export TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct
export ROOT=./data/Amazon/${CATEGORY}
export MONO_CUDA=0
export CUDA_VISIBLE=0,1,2,3
```

### 8.1 Stage 2 — SASRec collaborative embeddings (collab modes only)

```bash
bash scripts/rec_zoo/train_sasrec.sh \
  --data_dir "${ROOT}" \
  --epoch 10000 \
  --batch_size 128 \
  --hidden_factor 128 \
  --seq_size 50 \
  --lr 0.001 \
  --early_stop 50 \
  --cuda "${MONO_CUDA}" \
  --save_dir "./sasrec_models/${CATEGORY}"
```

Required output:

```
sasrec_models/{Category}/sasrec_item_emb.npy
```

This file is required by every `*_collab` fusion mode.

### 8.2 Stage 3 — multimodal item embeddings

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}" torchrun \
  --nproc_per_node=4 --master_port=29571 \
  rq/text2emb/multimodal_amazon_text2emb.py \
  --dataset "${CATEGORY}" \
  --root "${ROOT}" \
  --plm_checkpoint "${TEXT2EMB_MODEL}" \
  --plm_name Qwen3VL \
  --mode "${MODAL_CHOICE}" \
  --batch_size 4 \
  --max_sent_len 1024 \
  --moe_epochs 30 \
  --text_weight -1.0 \
  --collab_emb_path "./sasrec_models/${CATEGORY}/sasrec_item_emb.npy"
```

Output:

```
data/Amazon/{Category}/{Category}.emb-Qwen3VL-{MODAL_CHOICE}.npy
```

#### Fusion modes

| Mode | Description |
|------|-------------|
| `text_only` | Text-only embeddings |
| `image_only` | Image-only embeddings |
| `multimodal` | Concatenated text + image embeddings |
| `concat` | Concat text/image embeddings |
| `image2text` | VLM caption of the product image |
| `image2text_supp` | Caption + original text |
| `moe_decouple` | MoE decoupled text/image fusion |
| `contrastive_adapter` | Contrastive-adapter fusion |
| `multimodal_collab` | Multimodal + SASRec collaborative features |
| `concat_collab` | Concat + collaborative features |
| `moe_collab` | MoE + collaborative features |
| `csga_collab` | CSGA collaborative-graph aggregation |
| `csga_moe_collab` | CSGA → OMoE |
| `csga_moe_collab_v2` | CSGA → OMoE-v2 |

### 8.3 Stage 4 — RQ-VAE (4-level codebook)

```bash
CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/rqvae.py \
  --data_path "${ROOT}/${CATEGORY}.emb-Qwen3VL-${MODAL_CHOICE}.npy" \
  --ckpt_dir "./rq_output/${CATEGORY}" \
  --lr 1e-3 --epochs 5000 --batch_size 20480 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.0
```

Output:

```
rq_output/{Category}/best_collision_model.pth
```

### 8.4 Stage 5 — generate semantic ID indices

```bash
CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/generate_indices.py \
  --dataset "${CATEGORY}" \
  --ckpt_path "./rq_output/${CATEGORY}/best_collision_model.pth" \
  --output_dir "${ROOT}" \
  --device cuda:0
```

Optional low-frequency SID merging (collab modes):

```bash
CUDA_VISIBLE_DEVICES="${MONO_CUDA}" python rq/generate_indices.py \
  --dataset "${CATEGORY}" \
  --ckpt_path "./rq_output/${CATEGORY}/best_collision_model.pth" \
  --output_dir "${ROOT}" \
  --device cuda:0 \
  --collab_emb_path "./sasrec_models/${CATEGORY}/sasrec_item_emb.npy" \
  --train_inter "${ROOT}/${CATEGORY}.train.inter" \
  --merge_ratio 0.15
```

Output:

```
data/Amazon/{Category}/{Category}.index.json
```

### 8.5 Stage 6 — build SFT / RL datasets

```bash
python -m samgr.preprocess \
  --data_dir ./data/Amazon \
  --category "${CATEGORY}" \
  --output_dir "./data/${CATEGORY}" \
  --seq_sample -1 \
  --seed 42 \
  --data_source "${CATEGORY}" \
  --sid_num 4 \
  --only_task1 True
```

Output:

```
data/{Category}/
├── new_tokens.json
├── id2sid.json
├── sft/{train,valid,test}.json
└── rl/{train,valid,test}.json
```

Omit `--only_task1 True` to generate all tasks (task1–task5).

---

## 9. One-click end-to-end reproduction

`workflows/main.sh` runs stages 1–7 (stage 7 = SFT + evaluation). The most
important environment variables are:

| Variable | Default | Description |
|----------|---------|-------------|
| `CATEGORY` | `Industrial_and_Scientific` | Dataset name |
| `MODAL_CHOICE` | `text_only` | Fusion mode |
| `CUDA_VISIBLE` | `0,1,2,3` | GPUs for embedding / SFT |
| `MONO_CUDA` | `0` | GPU for RQ-VAE / index generation |
| `TEXT2EMB_MODEL` | local Qwen3-VL path | VLM checkpoint |
| `Optimization` | `dsz2` | DeepSpeed stage (`dsz0`/`dsz2`/`dsz3`) |
| `START_STAGE` | `1` | Resume from stage N (1–7) |
| `ONLY_TASK1` | `0` | `1` = task1-only ablation data |
| `SID_MERGE_RATIO` | `0` | Merge ratio for low-frequency items |
| `USER_K` / `ITEM_K` | `5` / `5` | k-core thresholds |
| `SAMPLE_SIZE` | `-1` | Limit users (`-1` = all) |
| `MOE_EPOCHS` | `30` | CSGA / OMoE training epochs |
| `EMB_BATCH_SIZE` | `4` | Embedding batch size |
| `SASREC_EPOCHS` | `10000` | SASRec training epochs |
| `SASREC_BATCH_SIZE` | `128` | SASRec batch size |
| `SASREC_HIDDEN` | `128` | SASRec hidden size |
| `SASREC_SEQ` | `50` | SASRec sequence length |
| `POSSESS_DATA_OUTPATH` | `./data/Amazon` | Stage-1 output root |
| `OUTPUT_DIR` | `./data/${CATEGORY}` | Stage-6 output root |
| `RQVAE_DIR` | `./rq_output/${CATEGORY}` | RQ-VAE checkpoint directory |

### 9.1 Text-only baseline

```bash
CATEGORY=Industrial_and_Scientific \
MODAL_CHOICE=text_only \
ONLY_TASK1=1 \
bash workflows/main.sh
```

### 9.2 Full CSGA → OMoE-v2 pipeline

```bash
CATEGORY=Industrial_and_Scientific \
MODAL_CHOICE=csga_moe_collab_v2 \
SID_MERGE_RATIO=0.15 \
Optimization=dsz2_csga_omoe_v2 \
TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct \
bash workflows/main.sh
```

### 9.3 Background execution

```bash
bash workflows/nohup_run.sh run pipeline bash workflows/main.sh
bash workflows/nohup_run.sh status pipeline
bash workflows/nohup_run.sh tail pipeline
bash workflows/nohup_run.sh stop pipeline
```

Logs go to `nohup_logs/`; pid/session metadata is stored alongside them.

### 9.4 Resume a pipeline

Every stage validates its expected artifacts. To resume:

```bash
START_STAGE=5 \
CATEGORY=Industrial_and_Scientific \
MODAL_CHOICE=csga_moe_collab_v2 \
bash workflows/main.sh
```

### 9.5 Sequential data → SFT → RL → eval

`workflows/run_all.sh` uses `workflows/run_config.sh` as the single source of
truth:

```bash
bash workflows/run_all.sh
```

Switch stages with:

```bash
RUN_DATA_PREP=1 RUN_SFT=1 RUN_RL=1 RUN_EVAL=1 bash workflows/run_all.sh
QUICK_TEST=1 bash workflows/run_all.sh   # tiny smoke run
```

---

## 10. SFT

SFT is launched through `workflows/sft.sh` with a YAML config from
`examples/train_full/{Category}/`.

```bash
CATEGORY=Industrial_and_Scientific \
Optimization=dsz2 \
SFT_YAML=examples/train_full/Industrial_and_Scientific/new_industry_rec_full_sft_0.5b_dsz2.yaml \
bash workflows/sft.sh
```

Equivalent direct command:

```bash
llamafactory-cli train examples/train_full/Industrial_and_Scientific/new_industry_rec_full_sft_0.5b_dsz2.yaml
```

Important YAML fields:

```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
add_tokens_list: 'data/Industrial_and_Scientific/new_tokens.json'
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json
dataset: Industrial_and_Scientific_train
eval_dataset: Industrial_and_Scientific_valid
output_dir: newsaves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz2
```

Make sure the dataset names exist in `data/dataset_info.json` and that
`add_tokens_list` points to the `new_tokens.json` produced in stage 6.

### Optional: codebook prior injection

```bash
python -m samgr.inject_priors \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --new_tokens_path data/Industrial_and_Scientific/new_tokens.json \
  --rqvae_ckpt rq_output/Industrial_and_Scientific/best_collision_model.pth \
  --output_path saves/Industrial_and_Scientific-prior \
  --blend_ratio 0.5
```

---

## 11. RL with GRPO

Edit `workflows/trl_trainer.sh` or override its environment variables, then run:

```bash
bash workflows/trl_trainer.sh
```

Or invoke the module directly through Accelerate:

```bash
accelerate launch \
  --config_file config/zero2.yaml \
  --num_processes 4 \
  --main_process_port 29503 \
  -m samgr.train_rl \
  --model newsaves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz2 \
  --data_dir data/Industrial_and_Scientific/rl \
  --index_path data/Industrial_and_Scientific/id2sid.json \
  --output_dir rl_outputs/Industrial_and_Scientific-qwen2.5-0.5b-instruct-grpo \
  --num_beams 16 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --max_completion_length 128 \
  --beta 1e-3 \
  --report_to wandb
```

Notes:

- `--index_path` must point to the `id2sid.json` produced in stage 6.
- `--data_dir` must contain `train.json`, `valid.json`, `test.json`.
- `--save_total_limit 1` is recommended to control checkpoint disk usage.

---

## 12. Evaluation

### 12.1 Standard evaluation

```bash
CATEGORY=Industrial_and_Scientific \
Optimization=dsz2 \
EVAL_EXP_NAME=newsaves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz2 \
EVAL_CUDA_LIST="0 1 2 3" \
bash workflows/evaluate.sh
```

The script:

1. Checks test data and `id2sid.json`.
2. Splits test JSON across GPUs with `samgr.split_json`.
3. Runs trie-constrained beam search on each GPU.
4. Merges predictions with `samgr.merge_json`.
5. Computes HR / NDCG metrics.

Outputs:

```
results/<exp_name>/final_result.json
results/<exp_name>/metrics.json
```

Key evaluation parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_EXP_NAME` | `newsaves/...` | Model checkpoint to evaluate |
| `EVAL_CUDA_LIST` | `0 1 2 3` | GPUs used for parallel evaluation |
| `EVAL_BATCH_SIZE` | `8` | Per-GPU batch size |
| `OUTPUT_DIR` | `data/{Category}` | Where SFT/RL data and `id2sid.json` live |

### 12.2 Determinism check

```bash
python -m samgr.check_seed \
  --model_name_or_path newsaves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz2 \
  --test_data_path data/Industrial_and_Scientific/sft/test.json \
  --index_path data/Industrial_and_Scientific/id2sid.json \
  --num_beams 20
```

Expected:

```
deterministic = True
```

---

## 13. Baselines

```bash
bash scripts/rec_zoo/train_gru.sh   --data_dir "data/Amazon/${CATEGORY}"
bash scripts/rec_zoo/train_caser.sh --data_dir "data/Amazon/${CATEGORY}"
bash scripts/rec_zoo/train_sasrec.sh --data_dir "data/Amazon/${CATEGORY}" \
  --save_dir "sasrec_models/${CATEGORY}"
```

SASRec additionally exports `sasrec_item_emb.npy`, required by all `*_collab`
fusion modes.

TIGER-style SID training helpers:

```bash
bash scripts/tiger/run_tiger.sh
bash scripts/tiger/run_tiger_eval.sh
```

---

## 14. Ablations and analysis

Analysis code and reports live under `analysis/`:

| Directory | Contents |
|-----------|----------|
| `analysis/omoe_ablation/` | OMoE component ablations and downstream retraining |
| `analysis/omoe_load_balance/` | Load-balancing and intervention experiments |
| `analysis/omoe_ind/` | Industrial dataset OMoE analysis |
| `analysis/omoe_sports10/` | Sports 10-core OMoE analysis |
| `analysis/samgr_sports10/` | SAM-GR fusion evidence, CSGA neighborhoods, CKA |
| `analysis/sid_collision_sports10/` | SID collision and TIGER beam analysis |

Run the OMoE ablation driver with a variant and stage list:

```bash
python analysis/omoe_ablation/run_downstream.py \
  --variant 01_full_omoe \
  --stages rq,index,preprocess,sft,eval,collect \
  --gpus 0,1,2,3 --eval-gpus 0,1,2,3
```

Reports are in each directory as `report.md` / `REPORT.md`. Large result
directories, figures and checkpoints are gitignored.

---

## 15. Generated artifacts

| Path | Produced by | Tracked? |
|------|-------------|----------|
| `data/Amazon/{Category}/` | stage 1–5 | No |
| `data/{Category}/sft`, `data/{Category}/rl` | stage 6 | No |
| `rq_output/{Category}/` | stage 4 | No |
| `sasrec_models/{Category}/` | stage 2 / baselines | No |
| `newsaves/`, `saves/` | SFT | No |
| `rl_outputs/` | RL | No |
| `results/` | evaluation | No |
| `runs/`, `experiments_new/`, `experiments_strong/` | experiments | No |
| `wandb/`, `logs/`, `nohup_logs/`, `temp/` | runtime | No |

The `.gitignore` excludes caches, environments, logs, temporary data,
checkpoints, embeddings, datasets, and analysis artifacts. Source code, configs,
docs, and small metadata remain tracked.

---

## 16. Troubleshooting

### Missing files in a resumed stage

Every stage checks its expected artifact. Read the error and confirm the
corresponding output exists (see §15). Use `START_STAGE=N` to rerun from an
earlier stage.

### CUDA / NCCL hangs

`workflows/main.sh` sets conservative NCCL defaults for the embedding stage.
If you have NVLink / InfiniBand, tune:

```bash
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_P2P_DISABLE=1     # only if P2P causes hangs
# export NCCL_IB_DISABLE=1      # only if IB is unavailable
```

### Out-of-memory

- Reduce `EMB_BATCH_SIZE` (embedding).
- Reduce `SFT_PER_DEVICE_BATCH` / `gradient_accumulation_steps` in the SFT YAML.
- Reduce `per_device_train_batch_size` in RL.
- Use DeepSpeed ZeRO-2 or ZeRO-3 (`Optimization=dsz2`/`dsz3`).

### Version incompatibilities

Align versions with §4. `DISABLE_VERSION_CHECK=1` is set by the workflow scripts
to avoid non-fatal LLaMA-Factory version warnings.

### `samgr` module not found

Run commands from the repository root, or install the project:

```bash
pip install -e .
```

All `python -m samgr.*` commands assume the repository root is the working
directory (the workflow scripts `cd` there automatically).

### Smoke test fails

Run each part manually:

```bash
python tests/make_smoke_data.py --out-dir temp/debug_raw --category SmokeTest
python -m samgr.split_json --help
python -m samgr.merge_json --help
bash -n workflows/main.sh
```

Check that `SMOKE_REAL_PYTHON` points to a working interpreter with PyTorch,
Transformers and Fire installed.

---

## 17. Reproducibility checklist

For a full result reproduction:

1. **Environment** matches §4; record `pip freeze` and GPU driver versions.
2. **Seeds**: stage 6 uses `--seed 42`; SFT/RL YAMLs set the generator seed;
   evaluation uses `seed=42` (or `42 + split_index` for multi-GPU).
3. **Data**: use the same raw files and k-core thresholds (`USER_K`, `ITEM_K`).
4. **Embedding mode**: record `MODAL_CHOICE`, `MOE_EPOCHS`, `TEXT_WEIGHT`,
   `COMBINE_ORIG_TEXT`, `SID_MERGE_RATIO`.
5. **RQ-VAE**: record `num_emb_list 256 256 256 256`, epochs, batch size,
   collision checkpoint.
6. **SFT**: record the exact YAML, model size, DeepSpeed stage, batch size,
   learning rate, epochs, and `new_tokens.json`.
7. **RL**: record `num_beams`, `beta`, learning rate, epochs, and dataset paths.
8. **Evaluation**: record beams (`50`), `max_new_tokens` (`256`), batch size,
   GPU split, and test set.
9. **Archive metrics** from `results/.../metrics.json` together with the command
   line / environment snapshot.
10. **Run the smoke test** (`bash tests/smoke_test.sh`) after any code or path
    change before launching a long job.

Minimal full-run recipe:

```bash
export CATEGORY=Industrial_and_Scientific
export MODAL_CHOICE=csga_moe_collab_v2
export TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct
export Optimization=dsz2_csga_omoe_v2
export SID_MERGE_RATIO=0.15

bash workflows/main.sh
```

---

## 18. License and acknowledgments

- [MIMIGenRec](https://github.com/cocoshe/MIMIGenRec) — base generative recommendation framework
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT framework
- [TRL](https://github.com/huggingface/trl) — reinforcement learning
- [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) — open generative recommendation baseline

Apache-2.0, same as the inherited MIMIGenRec codebase. See [LICENSE](LICENSE).
