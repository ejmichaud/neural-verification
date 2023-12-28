import os
import subprocess
from tqdm import tqdm

def run_create_dataset_scripts(main_folder):
    failed_scripts = []

    # Iterating through subfolders in the main folder
    for task_folder in os.listdir(main_folder):
        task_path = os.path.join(main_folder, task_folder)
        
        # Check if it's a directory
        if os.path.isdir(task_path):
            script_path = os.path.join(task_path, "create_dataset.py")
            
            # Check if the create_dataset.py script exists in this subfolder
            if os.path.isfile(script_path):
                try:
                    print(f"Running {script_path}...")
                    subprocess.run(["python", script_path], check=True)
                except subprocess.CalledProcessError:
                    failed_scripts.append(script_path)

    if failed_scripts:
        print("\nThe following scripts failed to run:")
        for script in failed_scripts:
            print(script)

folder_path = "/home/gridsan/vlad/meng_work/neural-verification/tasks"  # Replace with the path to your scripts
run_create_dataset_scripts(folder_path)
