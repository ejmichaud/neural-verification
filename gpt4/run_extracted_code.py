import os
import subprocess
import csv

def run_script_in_directories(base_path):
    with open('task_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task Name', 'Result'])  # Write header

        # Iterate through each task directory in the base path
        for task_folder in os.listdir(base_path):
            if task_folder.startswith('.'):
                continue
            task_folder_path = os.path.join(base_path, task_folder)

            # Check if it's a directory and not a file
            if os.path.isdir(task_folder_path):
                task_result = '0'  # Default to 'Failure'

                # Iterate through each iteration directory within the task directory
                for iteration_folder in os.listdir(task_folder_path):
                    iteration_folder_path = os.path.join(task_folder_path, iteration_folder)
                    if os.path.isdir(iteration_folder_path):
                        # Construct the script name based on the iteration folder name
                        script_name = f'extracted_code_{iteration_folder}.py'
                        script_path = os.path.join(iteration_folder_path, script_name)

                        # Check if the script exists
                        if not os.path.exists(script_path):
                            print(f"Script not found: {script_path}")
                            continue

                        # Change the current working directory to the iteration folder
                        os.chdir(iteration_folder_path)
                        # Run the script for the iteration
                        try:
                            process = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            output = process.stdout
                            # Check for success
                            if "success" in output.lower():
                                task_result = '1'
                                break  # Stop checking other iterations if one is successful
                        except subprocess.CalledProcessError as e:
                            print(f"An error occurred in {iteration_folder_path}: {e}")

                writer.writerow([task_folder, task_result])
                print(f"Processed task: {task_folder} - Result: {task_result}")

    print("Task processing complete. Results written to task_results.csv.")

# Example usage
base_path = os.getcwd()
run_script_in_directories(base_path)
