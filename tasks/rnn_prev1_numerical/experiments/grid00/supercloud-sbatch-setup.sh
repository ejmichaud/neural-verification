#!/bin/bash
#SBATCH --job-name=grid00
#SBATCH -o tasks/rnn_prev1_numerical_sc/experiments/grid00/logs/slurm-%A_%a.out
#SBATCH --error=tasks/rnn_prev1_numerical_sc/experiments/grid00/logs/slurm-%A_%a.err
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
echo $1
module load anaconda/2023a
pip install -e neural_verification
python tasks/rnn_prev1_numerical_sc/experiments/grid00/eval.py $1
