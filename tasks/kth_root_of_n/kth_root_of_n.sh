#!/bin/bash
#SBATCH -o tasks/kth_root_of_n/log/%j-run.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

module load anaconda/2023a
pip install -e neural_verification
python tasks/kth_root_of_n/train.py