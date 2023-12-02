import subprocess
import csv
from tqdm import tqdm

# Define the list of tasks
tasks = [
    "rnn_identity_numerical",
    "rnn_prev1_numerical",
    "rnn_prev2_numerical",
    "rnn_prev3_numerical",
    "rnn_prev4_numerical",
    "rnn_sum_numerical",
    "rnn_sum_last2_numerical",
    "rnn_sum_last3_numerical",
    "rnn_sum_last4_numerical",
    "rnn_or_last2_numerical",
    "rnn_parity_last2_numerical",
    "rnn_parity_last3_numerical",
    "rnn_parity_last4_numerical",
]


def run_task(task_name):
    """Run a task and return True if it succeeds, False otherwise."""
    try:
        # Run the task
        result = subprocess.run(["python", "auto_encode_RNN.py", "--task", task_name, "--device", "cpu"], capture_output=True, text=True)
        print(result)
    except Exception as e:
        print(f"Error running task {task_name}: {e}")
        return False

# Open a CSV file for writing
with open("task_status.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(["Task"])
    
    # Process each task
    for task in tqdm(tasks):
        writer.writerow([task])
        
for task in tqdm(tasks):
    run_task(task)
    

