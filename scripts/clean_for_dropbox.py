
import os
import sys
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to where the runs and results are saved.")
    parser.add_argument('--tasks_dir', type=str, 
        default="/om2/user/ericjm/neural-verification/tasks",
        help="Path to the directory containing the tasks (has folders with `create_dataset.py` scripts inside)")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    save_dir_name = os.path.basename(save_dir)

    results = pd.read_csv(os.path.join(save_dir, 'results.csv'))

    successes = results[results['status'] == 'success']

    output_dir = os.path.join(save_dir, f'{save_dir_name}_dropbox')
    os.mkdir(output_dir)

    # iterate through the rows
    for i in range(len(successes)):
        row = successes.iloc[i]
        # get the save_dir
        task_name = row['task']
        task_dir = row['save_dir']
        # create a directory for the task in the output_dir
        task_output_dir = os.path.join(output_dir, task_name)
        os.mkdir(task_output_dir)
        # copy all the files from task_dir to task_output_dir
        os.system(f'cp -r {task_dir}/* {task_output_dir}/')

        # also copy create_dataset.py from tasks_dir
        os.system(f'cp {os.path.join(args.tasks_dir, task_name, "create_dataset.py")} {task_output_dir}/')
