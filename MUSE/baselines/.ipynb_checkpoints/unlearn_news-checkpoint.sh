#!/bin/bash
#SBATCH -p mhong                # Partition name
#SBATCH --job-name=unlearn_news # Job name
#SBATCH --output=unlearn_news.out # Standard output file
#SBATCH --error=unlearn_news.err  # Standard error file
#SBATCH --gres=gpu:8            # Request 1 GPU
#SBATCH --time=24:00:00         # Job time limit (1 hour)
#SBATCH --nodes=1               # Request 2 nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --mem=100G               # Memory request (adjust as needed)


ALGO="minimax_npo_gdr"
CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
TOKENIZER_DIR="meta-llama/Llama-2-7b-hf"
MAX_LEN=2048
EPOCHS=10
LR='1e-5'
OUT_BASE="/projects/standard/mhong/shared/hadir/out_dir/$CORPUS"
PER_DEVICE_BATCH_SIZE=4   # 8 GPUs
#─────────────────────────────────────────────────────────────────────────
PROBE_LAYERS="8 10 12 14"
PROBE_LR='5e-5'             
PROBE_INNER_STEPS=3         
PROBE_BETA='0.5'           # weight of probe loss term in full objective
#─────────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python unlearn.py \
    --algo $ALGO \
    --model_dir $TARGET_DIR \
    --tokenizer_dir $TOKENIZER_DIR \
    --data_file $FORGET \
    --retain_data_file $RETAIN \
    --out_dir "$OUT_BASE/$ALGO" \
    --max_len $MAX_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --probe_layers $PROBE_LAYERS \
    --probe_lr $PROBE_LR \
    --probe_inner_steps $PROBE_INNER_STEPS \
    --probe_beta $PROBE_BETA