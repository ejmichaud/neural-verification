import os
import subprocess
import csv

def run_script_in_directories(base_path):
    with open('task_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task Name', 'Result'])  # Write header

        # Iterate through each subdirectory in the base path
        for folder in os.listdir(base_path):
            if folder.startswith('.'):
                continue
            folder_path = os.path.join(base_path, folder)

            # Check if it's a directory and not a file
            if os.path.isdir(folder_path):
                # Change the current working directory to the folder
                os.chdir(folder_path)
                extract_result = 'Error'
                # Run the 'create_dataset.py' script
                try:
                    process = subprocess.run(['python', 'extracted_code.py'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    output = process.stdout
                    if "failure" in output.lower() or "failed" in output.lower() or "fail" in output.lower():
                        extract_result = 'Failure'
                    else:
                        extract_result = 'Success'
                    print(f"Script completed in {folder_path}")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred in {folder_path}: {e}")
                    extract_result = 'Error'

                writer.writerow([folder, extract_result])

    print("Task processing complete. Results written to task_results.csv.")

# Example usage
base_path = '/home/gridsan/vlad/meng_work/neural-verification/gpt3/'
run_script_in_directories(base_path)
