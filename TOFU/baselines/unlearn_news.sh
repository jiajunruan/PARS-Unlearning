#!/bin/bash
#SBATCH -p mhong                # Partition name
#SBATCH --job-name=minmax_unlearn_tofu # Job name
#SBATCH --output=unlearn_news.out # Standard output file
#SBATCH --error=unlearn_news.err  # Standard error file
#SBATCH --gres=gpu:a100:2           # Request 1 GPU
#SBATCH --time=24:00:00         # Job time limit (1 hour)
#SBATCH --nodes=1               # Request 2 nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --mem=100G               # Memory request (adjust as needed)

module load cuda/12.1.1

python unlearn.py