# SAM-GR

Multimodal generative recommendation with collaborative-semantic aggregation (CSGA)
and Mixture-of-Experts (OMoE) fusion.

## Pipeline

```text
raw reviews + metadata + images
        -> item/interaction files
        -> (optional) SASRec collaborative embeddings
        -> multimodal item embeddings (text + image, optional CSGA/OMoE)
        -> RQ-VAE semantic IDs
        -> SID index
        -> SFT/RL datasets
        -> SFT -> (optional) GRPO -> evaluation
```

## Repository layout

```text
samgr/       Core Python modules and CLI entry points
workflows/   Shell orchestration (main.sh, sft.sh, evaluate.sh, ...)
rq/          RQ-VAE, SID generation, text2emb, CSGA, OMoE
scripts/     Baselines (rec_zoo), TIGER, diagnostics
rewards/     GRPO ranking rewards
data/        Data scripts and datasets
examples/    SFT YAML configs
config/      DeepSpeed configs for RL
tests/       Smoke test and stubs
analysis/    Analysis scripts and reports
```

## Requirements

- Python 3.12
- PyTorch, Transformers, TRL, Accelerate, PEFT, Datasets
- A Qwen3-VL checkpoint for item embeddings
- A Qwen2.5 checkpoint for the generative recommender

## Installation

```bash
git clone git@github.com:Echolarity/SAM-GR-official.git
cd SAM-GR
conda create -n samgr python=3.12 -y
conda activate samgr
pip install -e .
```

Verify:

```bash
python -c "import samgr.evaluate, samgr.preprocess, samgr.train_rl; print('OK')"
```

## Data layout

```text
data/
├── {Category}.jsonl.gz          # reviews
├── meta_{Category}.jsonl.gz     # metadata
└── {Category}_images/           # {asin}_MAIN.jpg
```

## Key variables

```bash
export CATEGORY=Industrial_and_Scientific
export MODAL_CHOICE=csga_moe_collab_v2
export TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

Useful switches:

- `START_STAGE=N` — resume from stage N.
- `ONLY_TASK1=1` — build task1-only data.
- `SID_MERGE_RATIO=0.15` — merge low-frequency SIDs into popular neighbors.
- `Optimization=dsz2` — DeepSpeed stage used by the SFT YAML.

## Reproduction

### 1. Build item and interaction files

```bash
python data/possess_data.py \
  --category "${CATEGORY}" \
  --sample_size -1 \
  --user_k 5 --item_k 5 \
  --output_path ./data/Amazon \
  --raw_dir ./data \
  --img_dir "./data/${CATEGORY}_images"
```

Output: `data/Amazon/${CATEGORY}/${CATEGORY}.item.json` and `*.inter` files.

### 2. SASRec embeddings (collab modes only)

```bash
bash scripts/rec_zoo/train_sasrec.sh \
  --data_dir "./data/Amazon/${CATEGORY}" \
  --epoch 10000 --batch_size 128 \
  --hidden_factor 128 --seq_size 50 --lr 0.001 --early_stop 50 \
  --cuda 0 \
  --save_dir "./sasrec_models/${CATEGORY}"
```

Output: `sasrec_models/${CATEGORY}/sasrec_item_emb.npy`.

### 3. Multimodal item embeddings

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 --master_port=29571 \
  rq/text2emb/multimodal_amazon_text2emb.py \
  --dataset "${CATEGORY}" \
  --root "./data/Amazon/${CATEGORY}" \
  --plm_checkpoint "${TEXT2EMB_MODEL}" \
  --plm_name Qwen3VL \
  --mode "${MODAL_CHOICE}" \
  --batch_size 4 --max_sent_len 1024 --moe_epochs 30 \
  --collab_emb_path "./sasrec_models/${CATEGORY}/sasrec_item_emb.npy"
```

Output: `data/Amazon/${CATEGORY}/${CATEGORY}.emb-Qwen3VL-${MODAL_CHOICE}.npy`.

### 4. RQ-VAE

```bash
python rq/rqvae.py \
  --data_path "./data/Amazon/${CATEGORY}/${CATEGORY}.emb-Qwen3VL-${MODAL_CHOICE}.npy" \
  --ckpt_dir "./rq_output/${CATEGORY}" \
  --lr 1e-3 --epochs 5000 --batch_size 20480 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.0
```

Output: `rq_output/${CATEGORY}/best_collision_model.pth`.

### 5. Generate SID indices

```bash
python rq/generate_indices.py \
  --dataset "${CATEGORY}" \
  --ckpt_path "./rq_output/${CATEGORY}/best_collision_model.pth" \
  --output_dir "./data/Amazon/${CATEGORY}" \
  --device cuda:0 \
  --collab_emb_path "./sasrec_models/${CATEGORY}/sasrec_item_emb.npy" \
  --train_inter "./data/Amazon/${CATEGORY}/${CATEGORY}.train.inter" \
  --merge_ratio 0.15
```

Omit the last three arguments to disable SID merging.
Output: `data/Amazon/${CATEGORY}/${CATEGORY}.index.json`.

### 6. Build SFT/RL datasets

```bash
python -m samgr.preprocess \
  --data_dir ./data/Amazon \
  --category "${CATEGORY}" \
  --output_dir "./data/${CATEGORY}" \
  --seq_sample -1 --seed 42 \
  --data_source "${CATEGORY}" \
  --sid_num 4 \
  --only_task1 True
```

Output: `data/${CATEGORY}/{new_tokens.json,id2sid.json,sft/,rl/}`.

### 7. SFT

```bash
CATEGORY="${CATEGORY}" \
Optimization=dsz2 \
bash workflows/sft.sh
```

Set `SFT_YAML` to use a different config. Output goes to the `output_dir`
declared in the YAML (usually under `newsaves/`).

### 8. RL (optional)

```bash
bash workflows/trl_trainer.sh
```

Edit the paths and hyper-parameters at the top of `workflows/trl_trainer.sh`
before running.

### 9. Evaluation

```bash
CATEGORY="${CATEGORY}" \
Optimization=dsz2 \
EVAL_EXP_NAME="newsaves/qwen2.5-0.5b/full/${CATEGORY}-sft-dsz2" \
bash workflows/evaluate.sh
```

Output: `results/<exp_name>/final_result.json` and `metrics.json`.

## One-click run

`workflows/main.sh` runs stages 1–7:

```bash
CATEGORY=Industrial_and_Scientific \
MODAL_CHOICE=csga_moe_collab_v2 \
TEXT2EMB_MODEL=/path/to/Qwen3-VL-8B-Instruct \
bash workflows/main.sh
```

Background execution:

```bash
bash workflows/nohup_run.sh run pipeline bash workflows/main.sh
bash workflows/nohup_run.sh status pipeline
bash workflows/nohup_run.sh tail pipeline
bash workflows/nohup_run.sh stop pipeline
```

Resume:

```bash
START_STAGE=5 bash workflows/main.sh
```

## Fusion modes

```text
text_only  image_only  multimodal  concat  image2text  image2text_supp
moe_decouple  contrastive_adapter  multimodal_collab  concat_collab
moe_collab  csga_collab  csga_moe_collab  csga_moe_collab_v2
```

## Smoke test

```bash
bash tests/smoke_test.sh
```

Expected final line:

```text
SMOKE TEST PASSED
```

The test runs the real data, preprocessing and split/merge code on synthetic
data, and stubs only GPU/model-heavy stages so it completes in seconds.

## Outputs

| Path | Produced by |
|------|-------------|
| `data/Amazon/{Category}/` | stages 1–5 |
| `data/{Category}/sft`, `rl` | stage 6 |
| `rq_output/{Category}/` | stage 4 |
| `sasrec_models/{Category}/` | stage 2 |
| `newsaves/`, `saves/` | SFT |
| `rl_outputs/` | RL |
| `results/` | evaluation |

## Troubleshooting

- **Missing artifact on resume** — check the stage output above, then rerun from
  an earlier `START_STAGE`.
- **Out of memory** — lower `EMB_BATCH_SIZE`, the SFT batch size, or the RL batch
  size; use `Optimization=dsz2`/`dsz3`.
- **NCCL hangs** — set `NCCL_DEBUG=INFO` and, if needed,
  `NCCL_P2P_DISABLE=1` or `NCCL_IB_DISABLE=1`.

## License

Apache-2.0. See [LICENSE](LICENSE).
