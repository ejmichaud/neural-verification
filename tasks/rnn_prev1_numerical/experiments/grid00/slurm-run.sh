#!/bin/bash
#SBATCH --job-name=grid00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-0:20:00
#SBATCH --output=/om2/user/ericjm/neural-verification/tasks/rnn_prev1_numerical/experiments/grid00/logs/slurm-%A_%a.out
#SBATCH --error=/om2/user/ericjm/neural-verification/tasks/rnn_prev1_numerical/experiments/grid00/logs/slurm-%A_%a.err
#SBATCH --mem=8GB
#SBATCH --array=0-19

python /om2/user/ericjm/neural-verification/tasks/rnn_prev1_numerical/experiments/grid00/eval.py $SLURM_ARRAY_TASK_ID
