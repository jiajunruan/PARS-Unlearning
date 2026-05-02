#!/bin/bash
#SBATCH -p mhong                # Partition name
#SBATCH --job-name=unlearn_books # Job name
#SBATCH --output=unlearn_books.out # Standard output file
#SBATCH --error=unlearn_books.err  # Standard error file
#SBATCH --gres=gpu:8            # Request 1 GPU
#SBATCH --time=05:00:00         # Job time limit (1 hour)
#SBATCH --nodes=1               # Request 2 nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --mem=100G               # Memory request (adjust as needed)

algo="npo_gdr"
CORPUS="books"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="HadiUMN/Llama-2-7B-MUSE-Books-5lr"
#"muse-bench/MUSE-Books_target"
MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir 'meta-llama/Llama-2-7b-hf' \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "/projects/standard/mhong/shared/hadir/out_dir/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE