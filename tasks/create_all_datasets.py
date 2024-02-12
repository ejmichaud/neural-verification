import os
import subprocess

def run_create_dataset_scripts(main_folder):
    failed_scripts = []

    # Iterating through subfolders in the main folder
    for task_folder in os.listdir(main_folder):
        task_path = os.path.join(main_folder, task_folder)
        
        # Check if it's a directory
        if os.path.isdir(task_path):
            data_file_path = os.path.join(task_path, "data.pt")
            script_path = os.path.join(task_path, "create_dataset.py")
            
            # Check if the data.pt file exists, if so, skip this folder
            if os.path.isfile(data_file_path):
                print(f"Skipping {task_folder}, data.pt already exists.")
                continue

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

folder_path = "./"  # Replace with the path to your scripts
run_create_dataset_scripts(folder_path)
