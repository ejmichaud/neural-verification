#!/bin/bash

# Loop 20 times
for i in {1..20}
do
   sbatch tasks/rnn_unique3_numerical/experiments/grid00/supercloud-sbatch-setup.sh $i
   
done

# End of the script
