
"""
Takes in a config file specifying the tasks to run and parameters to use
for the architecture search on each task. The config also specifies the 
where to save the results of the search. We save a csv
where each row is a task and the columns include the details
of the best architecture, where the best architecture's run is saved at,
etc.

Let's have the config file be a yaml file

What this script produces:
    Within save_dir:
        * creates a directory with the current timestamp. this identifies the search
        * creates a directory for each task in the config file
        * performs architecture search and saves results each tasks's directory
        * saves a csv with the results of the search in the search's directory
        * saves a csv summarizing the results across all searches (all timestamps) in save_dir
"""

import os
import sys
import argparse
import yaml
import subprocess
import time

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --output={path}/slurm-%A.out
#SBATCH --error={path}/slurm-%A.err
#SBATCH --mem=8GB

echo "started" > {path}/status.txt
python {architecture_search_script} {args}
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
        help="Path to the config file specifying the tasks to run and their search args.")
    parser.add_argument('--tasks_dir', type=str, default="/om2/user/ericjm/neural-verification/tasks",
        help="Path to the directory containing the tasks (datasets or create_dataset.py scripts) to run.")
    parser.add_argument('--architecture_search_script', type=str, 
        default="/om2/user/ericjm/neural-verification/scripts/rnn_architecture_search.py") 
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save all runs and results.")
    args = parser.parse_args()
    # expand the paths to full absolute paths
    args.config = os.path.abspath(args.config)
    args.tasks_dir = os.path.abspath(args.tasks_dir)
    args.architecture_search_script = os.path.abspath(args.architecture_search_script)
    args.save_dir = os.path.abspath(args.save_dir)

    SEARCH_TIMESTAMP = str(int(time.time()))
    os.makedirs(os.path.join(args.save_dir, SEARCH_TIMESTAMP), exist_ok=True)
    with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, f'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # load up the config specifying the jobs and their search args
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    job_ids = {}

    for task, task_config in config.items():

        if not task_config['include']:
            continue # skip this task

        args_str = ""
        # see if the task has a dataset
        if not os.path.exists(os.path.join(args.tasks_dir, task, 'data.pt')):
            # run the create_dataset.py script
            os.system(f'python {os.path.join(args.tasks_dir, task, "create_dataset.py")}')
        args_str += f' --data {os.path.join(args.tasks_dir, task, "data.pt")}'
        task_args = task_config['args']
        for arg, val in task_args.items():
            if arg == 'vectorize_input':
                if val:
                    args_str += f' --{arg}'
            else:
                args_str += f' --{arg} {val}'
        args_str += f' --save_dir {os.path.join(args.save_dir, SEARCH_TIMESTAMP, task)}'
        args_str += '\n'

        slurm_script = SLURM_TEMPLATE.format(
            task=task,
            path=os.path.join(args.save_dir, SEARCH_TIMESTAMP, task),
            args=args_str,
            architecture_search_script=args.architecture_search_script
        )

        # create a directory for the task
        os.mkdir(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task))

        job_id = submit_slurm_job(slurm_script)
        job_ids[task] = job_id
        print(f"Submitted job for task {task} with ID: {job_id}")

    while True:
        time.sleep(5)
        n_started = 0
        n_errored = 0
        n_finished = 0
        # simply check the status of each job by looking at the status.txt file
        for task, job_id in job_ids.items():
            if os.path.exists(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'status.txt')):
                with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'status.txt'), 'r') as f:
                    status = f.read().strip()
                if status == 'finished':
                    n_finished += 1
                elif status == 'errored':
                    n_errored += 1
                elif status == 'started':
                    n_started += 1
                else:
                    print(f"ERROR: status.txt file for task {task} has unexpected contents: {status}")
        print(f"Started {max(n_started, n_finished)}/{len(job_ids)} jobs and finished {n_finished}/{len(job_ids)} jobs.")
        if n_finished + n_errored == len(job_ids):
            print(f"Exiting. Finished {n_finished} and errored {n_errored} jobs.")
            break

    # parse the results of the slurm jobs and save them to a csv
    results = []
    for task, job_id in job_ids.items():
        if os.path.exists(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'smallest_run_args.yaml')):
            with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, task, 'smallest_run_args.yaml'), 'r') as f:
                best_args = yaml.load(f, Loader=yaml.FullLoader)
            # save the info we care about to results for later saving to csv
            results.append(
                (   
                    task,
                    "success",
                    best_args['output_mlp_depth'],
                    best_args['hidden_mlp_depth'],
                    best_args['hidden_dim'],
                    best_args['output_mlp_width'],
                    best_args['hidden_mlp_width'],
                    best_args['input_dim'],
                    best_args['output_dim'],
                    best_args['loss_fn'],
                    best_args['vectorize_input'],
                    best_args['save_dir']
                )
            )
        else:
            results.append((task, "failure"))
    
    # save these to a dedicated CSV within the SEARCH_TIMESTAMP directory
    with open(os.path.join(args.save_dir, SEARCH_TIMESTAMP, 'results.csv'), 'w') as f:
        f.write('task,success,output_mlp_depth,hidden_mlp_depth,hidden_dim,output_mlp_width,hidden_mlp_width,input_dim,output_dim,loss_fn,vectorize_input,save_dir\n')
        for result in results:
            f.write(','.join([str(x) for x in result]) + '\n')
 
    # save results also to a summary csv directly in save_dir. make the first row the column names
    # if there already exists a summary.csv, override the rows
    # for the tasks that were just run
    if os.path.exists(os.path.join(args.save_dir, 'summary.csv')):
        with open(os.path.join(args.save_dir, 'summary.csv'), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(args.save_dir, 'summary.csv'), 'w') as f:
            f.write(lines[0])
            for line in lines[1:]:
                task = line.split(',')[0]
                if task not in job_ids:
                    f.write(line)
            for result in results:
                f.write(','.join([str(x) for x in result]) + '\n')
    else:
        with open(os.path.join(args.save_dir, 'summary.csv'), 'w') as f:
            f.write('task,success,output_mlp_depth,hidden_mlp_depth,hidden_dim,output_mlp_width,hidden_mlp_width,input_dim,output_dim,loss_fn,vectorize_input,save_dir\n')
            for result in results:
                f.write(','.join([str(x) for x in result]) + '\n')
