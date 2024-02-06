
"""
This script performs further training on networks which have been
already trained and hammered. Well, it doesn't do the training itself,
but runs a script which does the training on each slurm node.

There is a directory in the repository root directory called `hammered_models`
which have directories for each task and then a `model_hammered.pt` file
which contains the model state dict. To further train these models, we
train as normal except that we load the model state dict from the
`model_hammered.pt` file and then constrain the parameters whicih are 
exactly zero to remain zero.

To initialize the GeneralRNN correctly, we need the model config which
is in the `first_paper` directory in the repository root directory. This
directory contains a directory for each task which contains a `model_config.pt`
file which contains the model config.

We also need various values which specify how the network treats the dataset
(e.g. as one-hots or not). These can be found in the `first_paper` directory
in the repository root directory. This directory contains a directory for each
task which contains an `args.yaml` file which contains the arguments used
to train the network.

To get the datasets, we use the `tasks/` directory in the repository root
directory. This directory contains a directory for each task which contains
a `data.pt` file which contains the dataset.


trains a GeneralRNN on a given task. This script is pretty general.
It can be used with arbitrary combinations of: whether the input is a vector of 
real values or should be a one-hot vector, whether the output is treated
like a regression or classification problem, a vector of real values with 
mse/log loss or a vector of real values (logits) with cross-entropy loss.

For a typical regression problem with 1d real valued input and 1d real valued
output, use either --loss "mse" or --loss "log" and do not use --vectorize_input.
Note that the inputs and outputs can still be integers here, they are just treated
as real values.

For a typical classification problem with 1d integer input and 1d integer output,
use --loss "cross_entropy" and --vectorize_input.

If inputs are lists of vectors rather than lists of integers (such as in binary 
logical operators case where at each sequence position we have a 2d vector),
then do NOT use --vectorize_input, just specify the --input_dim as the dimension
of the vectors (e.g. 2). 
"""

import os
import random
import argparse
import json
import yaml

import subprocess
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --output={path}/slurm-%A.out
#SBATCH --error={path}/slurm-%A.err
#SBATCH --mem=8GB

conda activate phase-changes
echo "started" > {path}/status.txt
python {training_script} {args}
if [ $? -eq 0 ]; then
    echo "finished" > {path}/status.txt
else
    echo "errored" > {path}/status.txt
fi
"""

def submit_slurm_job(slurm_script):
    # Submit the job with sbatch
    result = subprocess.run(['sbatch'], input=slurm_script.encode(), stdout=subprocess.PIPE)
    
    # Parse the job ID from the output
    job_id = result.stdout.decode().split()[-1].strip()
    return job_id

if __name__ == '__main__':

    hammered_networks_dir = "/om2/user/ericjm/neural-verification/regularized_hammered_models"
    first_paper_dir = "/om2/user/ericjm/neural-verification/first_paper"
    tasks_dir = "/om2/user/ericjm/neural-verification/tasks"
    train_hammers_script = "/om2/user/ericjm/neural-verification/scripts/train_hammers.py"
    training_steps = 10000

    # iterate through the hammered networks
    # tasks = os.listdir(hammered_networks_dir)
    # actually tasks are only represented by directories
    tasks = [d for d in os.listdir(hammered_networks_dir) if os.path.isdir(os.path.join(hammered_networks_dir, d))]
    print(f"Found {len(tasks)} tasks that have been hammered.")
    for task in tasks:

        # see if the task has a dataset
        if not os.path.exists(os.path.join(tasks_dir, task, 'data.pt')):
            # run the create_dataset.py script
            print(f"Creating dataset for task {task}")
            os.system(f'python {os.path.join(tasks_dir, task, "create_dataset.py")}')

        args = ""
        args += f"--args {os.path.join(first_paper_dir, task, 'args.yaml')} "
        args += f"--model_config {os.path.join(first_paper_dir, task, 'model_config.pt')} "
        args += f"--save_dir {os.path.join(hammered_networks_dir, task)} "
        args += f"--steps {training_steps} "
        
        # submit a job to train the network
        slurm_script = SLURM_TEMPLATE.format(
            task=task, # this is just needed for the slurm job name
            path=os.path.join(hammered_networks_dir, task),
            args=args,
            training_script=train_hammers_script
        )

        job_id = submit_slurm_job(slurm_script)
        print(f"Submitted job for task {task} with ID: {job_id}")

        # let's change how the `train_hammers.py` script works, so that it 
        # just takes a path to a config, rather than a bunch of arguments
        # as strings

        # load up the args
        # with open(os.path.join(first_paper_dir, task, 'args.yaml'), 'r') as f:
        #     args = yaml.load(f, Loader=yaml.FullLoader)

        # override the save_dir
        # args.save_dir = os.path.join(hammered_networks_dir, task)

        # args_str = ""
        # # see if the task has a dataset
        # if not os.path.exists(os.path.join(tasks_dir, task, 'data.pt')):
        #     # run the create_dataset.py script
        #     print(f"Creating dataset for task {task}")
        #     os.system(f'python {os.path.join(tasks_dir, task, "create_dataset.py")}')
        # args_str += f' --data {os.path.join(tasks_dir, task, "data.pt")}'
        # task_args = task_config['args']
        # for arg, val in task_args.items():
        #     if arg == 'vectorize_input':
        #         if val:
        #             args_str += f' --{arg}'
        #     else:
        #         args_str += f' --{arg} {val}'
        # args_str += f' --save_dir {os.path.join(args.save_dir, SEARCH_TIMESTAMP, task)}'
        # args_str += '\n'



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True,
    #     help="Path to the config file specifying the tasks to run and their search args.")
    # parser.add_argument('--tasks_dir', type=str, default="/om2/user/ericjm/neural-verification/tasks",
    #     help="Path to the directory containing the tasks (datasets or create_dataset.py scripts) to run.")
    # parser.add_argument('--architecture_search_script', type=str, 
    #     default="/om2/user/ericjm/neural-verification/scripts/rnn_architecture_search.py") 
    # parser.add_argument('--save_dir', type=str, required=True, help="Directory to save all runs and results.")
    # args = parser.parse_args()
    # # expand the paths to full absolute paths
    # args.config = os.path.abspath(args.config)
    # args.tasks_dir = os.path.abspath(args.tasks_dir)
    # args.architecture_search_script = os.path.abspath(args.architecture_search_script)
    # args.save_dir = os.path.abspath(args.save_dir)

    # SEARCH_TIMESTAMP = str(int(time.time()))
    # os.makedirs(os.path.join(args.save_dir, SEARCH_TIMESTAMP), exist_ok=True)
    # with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, f'args.yaml'), 'w') as f:
    #     yaml.dump(vars(args), f)

    # # load up the config specifying the jobs and their search args
    # with open(args.config, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    # # save a copy of the config in the search directory
    # with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, 'search.yaml'), 'w') as f:
    #     yaml.dump(config, f)
    
    # job_ids = {}

    # for task, task_config in config.items():

    #     if not task_config['include']:
    #         continue # skip this task

    #     args_str = ""
    #     # see if the task has a dataset
    #     if not os.path.exists(os.path.join(args.tasks_dir, task, 'data.pt')):
    #         # run the create_dataset.py script
    #         os.system(f'python {os.path.join(args.tasks_dir, task, "create_dataset.py")}')
    #     args_str += f' --data {os.path.join(args.tasks_dir, task, "data.pt")}'
    #     task_args = task_config['args']
    #     for arg, val in task_args.items():
    #         if arg == 'vectorize_input':
    #             if val:
    #                 args_str += f' --{arg}'
    #         else:
    #             args_str += f' --{arg} {val}'
    #     args_str += f' --save_dir {os.path.join(args.save_dir, SEARCH_TIMESTAMP, task)}'
    #     args_str += '\n'

    #     slurm_script = SLURM_TEMPLATE.format(
    #         task=task, # this is just needed for the slurm job name
    #         path=os.path.join(args.save_dir, SEARCH_TIMESTAMP, task),
    #         args=args_str,
    #         architecture_search_script=args.architecture_search_script
    #     )



    #     # create a directory for the task
    #     os.mkdir(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task))

    #     job_id = submit_slurm_job(slurm_script)
    #     job_ids[task] = job_id
    #     print(f"Submitted job for task {task} with ID: {job_id}")

    # while True:
    #     time.sleep(30)
    #     statuses = {}
    #     # simply check the status of each job by looking at the status.txt file
    #     for task, job_id in job_ids.items():
    #         if os.path.exists(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'status.txt')):
    #             with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'status.txt'), 'r') as f:
    #                 status = f.read().strip()
    #             if status != statuses.get(task, None):
    #                 print(f"Task {task} has updated status: {status}")
    #             statuses[task] = status
    #     n_finished = sum([1 for status in statuses.values() if status == 'finished'])
    #     n_errored = sum([1 for status in statuses.values() if status == 'errored'])
    #     if n_finished + n_errored == len(job_ids):
    #         print(f"Exiting. Finished {n_finished} and errored {n_errored} jobs.")
    #         break



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Trains a GeneralRNN")
#     parser.add_argument('--data', type=str, help='Path to dataset, a .pt file')
#     parser.add_argument('--loss_fn', type=str, default="mse", help='Either "mse" or "log" or "cross_entropy"')
#     parser.add_argument('--vectorize_input', action="store_true", help="Convert input ints to one-hot vectors")
#     parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
#     parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
#     parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden layer dimension')
#     parser.add_argument('--hidden_mlp_depth', type=int, default=1, help='Depth of hidden MLP')
#     parser.add_argument('--hidden_mlp_width', type=int, default=16, help='Width of hidden MLP')
#     parser.add_argument('--output_mlp_depth', type=int, default=1, help='Depth of output MLP')
#     parser.add_argument('--output_mlp_width', type=int, default=16, help='Width of output MLP')
#     parser.add_argument('--activation', type=str, default="ReLU", help='Activation function')
#     parser.add_argument('--steps', type=int, default=10000, help='Number of steps for training')
#     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
#     parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
#     parser.add_argument('--train_batch_size', type=int, default=1024, help='Batch size for training')
#     parser.add_argument('--test_batch_size', type=int, default=8192, help='Batch size for testing')
#     parser.add_argument('--ignore_first_n_elements', type=int, default=0, help='Ignore loss at first n sequence positions')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     parser.add_argument('--dtype', type=str, default="float32", help='Pytorch default dtype')
#     # parser.add_argument("--slurm", action="store_true", help="Enable parallel training with slurm")
#     parser.add_argument('--save_dir', type=str, default="0", help='Directory to save results')
#     parser.add_argument('--progress_bar', action="store_true", help='Show progress bar during training')
#     args = parser.parse_args()
#     train(args)
