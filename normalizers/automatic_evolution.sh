#!/bin/bash
#SBATCH --job-name=automatic_evolution
#SBATCH --time=0-12:00:00
#SBATCH --output=automatic_evolution.out
#SBATCH --error=automatic_evolution.err

module load anaconda/2023b
pip install -e neural_verification
python automatic_evolution.py
