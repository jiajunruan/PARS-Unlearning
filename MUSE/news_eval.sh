#!/bin/bash
#SBATCH -p mhong                # Partition name
#SBATCH --job-name=unlearn_books_eval # Job name
#SBATCH --output=unlearn_books_eval.out # Standard output file
#SBATCH --error=unlearn_books.err  # Standard error file
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=07:00:00         # Job time limit (1 hour)
#SBATCH --nodes=1               # Request 2 nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --mem=40G               # Memory request (adjust as needed)

export CUDA_VISIBLE_DEVICES=7
python eval.py \
  --model_dirs "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-102" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-204" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-306" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-408" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-510" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-612" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-714" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-816" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-918" "/projects/standard/mhong/shared/hadir/out_dir/news/npo_gdr/checkpoint-1020" \
  --names "checkpoint-102" "checkpoint-204" "checkpoint-306" "checkpoint-408" "checkpoint-510" "checkpoint-612" "checkpoint-714" "checkpoint-816" "checkpoint-918" "checkpoint-1020" \
  --corpus "news" \
  --out_file "/users/2/hadir/Desktop/MUSE-NPO/result/out_npo_original.csv"



####--- #SBATCH --partition=interactive-gpu   

