

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

UPDATE:
    This script is a smaller version that just searches the files
    written by search_all.py and writes the results to a csv.
"""

import os
import sys
import argparse
import yaml
import subprocess
import time
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, 
        help="Directory where runs and results were saved. A unix timestamp.")
    args = parser.parse_args()
    # expand the paths to full absolute paths
    args.save_dir = os.path.abspath(args.save_dir)

    # load up the config that was used to run this search
    # it's `search.yaml` inside the save_dir
    with open(os.path.join(args.save_dir, 'search.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # only consider the configs with `include: true`
    config = {task: task_config for task, task_config in config.items() if task_config['include']}

    results = []
    for task, task_config in config.items():
        if os.path.exists(os.path.join(args.save_dir, task, 'smallest_run_args.yaml')):
            with open(os.path.join(args.save_dir, task, 'smallest_run_args.yaml'), 'r') as f:
                best_args = yaml.load(f, Loader=yaml.FullLoader)
            if best_args is None: # under what circumstances is this True?
                results.append((task, "error"))
                continue
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
    with open(os.path.join(args.save_dir, 'results.csv'), 'w') as f:
        f.write('task,status,output_mlp_depth,hidden_mlp_depth,hidden_dim,output_mlp_width,hidden_mlp_width,input_dim,output_dim,loss_fn,vectorize_input,save_dir\n')
        for result in results:
            f.write(','.join([str(x) for x in result]) + '\n')

    # save results also to a summary csv directly in save_dir. make the first row the column names
    # if there already exists a summary.csv, override the rows
    # for the tasks that were just run
    # if os.path.exists(os.path.join(args.save_dir, 'summary.csv')):
    #     with open(os.path.join(args.save_dir, 'summary.csv'), 'r') as f:
    #         lines = f.readlines()
    #     with open(os.path.join(args.save_dir, 'summary.csv'), 'w') as f:
    #         f.write(lines[0])
    #         for line in lines[1:]:
    #             task = line.split(',')[0]
    #             if task not in job_ids:
    #                 f.write(line)
    #         for result in results:
    #             f.write(','.join([str(x) for x in result]) + '\n')
    # else:
    #     with open(os.path.join(args.save_dir, 'summary.csv'), 'w') as f:
    #         f.write('task,success,output_mlp_depth,hidden_mlp_depth,hidden_dim,output_mlp_width,hidden_mlp_width,input_dim,output_dim,loss_fn,vectorize_input,save_dir\n')
    #         for result in results:
    #             f.write(','.join([str(x) for x in result]) + '\n')

