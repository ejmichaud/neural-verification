import subprocess
from tqdm import tqdm
import argparse

# Define the list of tasks
tasks = ['Binary_Addition',
 'Base_3_Addition',
 'Base_4_Addition',
 'Base_5_Addition',
 'Base_6_Addition',
 'Base_7_Addition',
 'Bitwise_Xor',
 'Bitwise_Or',
 'Bitwise_And',
 'Bitwise_Not',
 'Parity_Last2',
 'Parity_Last3',
 'Parity_Last4',
 'Parity_All',
 'Parity_Zeros',
 'Evens_Counter',
 'Sum_All',
 'Sum_Last2',
 'Sum_Last3',
 'Sum_Last4',
 'Sum_Last5',
 'Sum_Last6',
 'Sum_Last7',
 'Current_Number',
 'Prev1',
 'Prev2',
 'Prev3',
 'Prev4',
 'Prev5',
 'Previous_Equals_Current',
 'Diff_Last2',
 'Abs_Diff',
 'Abs_Current',
 'Diff_Abs_Values',
 'Min_Seen',
 'Max_Seen',
 'Majority_0_1',
 'Majority_0_2',
 'Majority_0_3',
 'Evens_Detector',
 'Perfect_Square_Detector',
 'Bit_Palindrome',
 'Balanced_Parenthesis',
 'Parity_Bits_Mod2',
 'Alternating_Last3',
 'Alternating_Last4',
 'Bit_Shift_Right',
 'Bit_Dot_Prod_Mod2',
 'Div_3',
 'Div_5',
 'Div_7',
 'Add_Mod_3',
 'Add_Mod_4',
 'Add_Mod_5',
 'Add_Mod_6',
 'Add_Mod_7',
 'Add_Mod_8',
 'Dithering',
 'Newton_Freebody',
 'Newton_Gravity',
 'Newton_Spring',
 'Newton_Magnetic']

# Import argparse and set up argument parsing
parser = argparse.ArgumentParser(description='Run tasks and extract code.')
parser.add_argument('--iterations', type=int, default=3, help='Number of iterations to run each task')
args = parser.parse_args()

def run_task_and_extract_code(task_name, iterations):
    for iteration in range(iterations):
        try:
            task_result = subprocess.run(["python", "prompt.py", "--task", task_name, "--iterations", str(iterations)], capture_output=True, text=True)
            # Rest of the code remains the same
        except Exception as e:
            print(f"Error running task {task_name}: {e}")

# Use the parsed iterations value
for task in tqdm(tasks):
    run_task_and_extract_code(task, args.iterations)

print("Task processing complete.")

