
import random
import time

from itertools import product
import os
import sys

import numpy as np

hidden_dims = [1, 2, 3]
seeds = list(range(5))
configs = list(product(hidden_dims, seeds)) # 10
# print(len(configs))
# exit()

if __name__ == '__main__':
    task_idx = int(sys.argv[1])
    hidden_dim, seed = configs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /home/gridsan/cguo/neural-verification/tasks/rnn_unique2_numerical/experiments/grid00/train.py \
                    --seed {seed} \
                    --steps 10000 \
                    --train_batch_size 4096 \
                    --test_batch_size 65536 \
                    --hidden_dim {hidden_dim} \
                    --hidden_mlp_depth 1 \
                    --output_mlp_depth 1 \
                    --progress_bar False \
                    --save_dir results/{task_idx} \
                    """)

