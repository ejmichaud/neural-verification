#!/bin/bash

# Loop 20 times
for i in {1..20}
do
   # Call sbatch command
   sbatch tasks/rnn_prev1_numerical_sc/experiments/grid00/sbatch-setup.sh $i
   
   # You can add any other commands or options to sbatch as needed
   # For example:
   # sbatch --job-name=test_job_$i your_job_script.sh
done

# End of the script
