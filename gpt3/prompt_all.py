import subprocess
import csv
from tqdm import tqdm

# Define the list of tasks
tasks = [
    "balanced_parentheses",
    "bit_addition",
    "rnn_abs_value_numerical",
    "rnn_abs_value_of_diff_numerical",
    "rnn_bitwise_and",
    "rnn_bitwise_not",
    "rnn_bitwise_or",
    "rnn_count_power_of_2s_bool",
    "rnn_diff_last2_numerical",
    "rnn_diff_of_abs_value_numerical",
    "rnn_identity_numerical",
    "rnn_majority0_1",
    "rnn_majority0_2",
    "rnn_majority0_3",
    "rnn_majority0_4",
    "rnn_majority0_5",
    "rnn_parity_last2_numerical",
    "rnn_parity_last3_numerical",
    "rnn_parity_last4_numerical",
    "rnn_parity_numerical",
    "rnn_prev1_numerical",
    "rnn_prev2_numerical",
    "rnn_prev3_numerical",
    "rnn_prev4_numerical",
    "rnn_prev7_numerical",
    "rnn_prev8_numerical",
    "rnn_sum_last2_numerical",
    "rnn_sum_last3_numerical",
    "rnn_sum_last4_numerical",
    "rnn_sum_last5_numerical",
    "rnn_sum_last6_numerical",
    "rnn_sum_last7_numerical",
    "rnn_sum_numerical",
    "rnn_unique2_numerical",
    "rnn_vowel_counter",
    "vowel_counter"
]

def run_task_and_extract_code(task_name):
    """Run a task, extract code, and return 'Success' or 'Failure'."""
    try:
        # Run the task
        task_result = subprocess.run(["python", "prompt.py", "--task", task_name], capture_output=True, text=True)
        print("STDOUT:", task_result.stdout)
        print("STDERR:", task_result.stderr)

        # Run extract_code.py and check for success
        extract_result = subprocess.run(["python", "extract_code.py"], capture_output=True, text=True)
    except Exception as e:
        print(f"Error running task {task_name}: {e}")
        

for task in tqdm(tasks):
    result = run_task_and_extract_code(task)
print("Task processing complete.")


