#!/bin/bash
#SBATCH --job-name=grid00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-0:20:00
#SBATCH --output=/home/gridsan/cguo/neural-verification/tasks/count_unique/logs/slurm-%A_%a.out
#SBATCH --error=/home/gridsan/cguo/neural-verification/tasks/count_unique/grid00/logs/slurm-%A_%a.err
#SBATCH --mem=8GB
#SBATCH --array=0-19
echo "this script doesn't work for supercloud right now"
python /home/gridsan/cguo/neural-verification/tasks/count_unique/eval.py $SLURM_ARRAY_TASK_ID
