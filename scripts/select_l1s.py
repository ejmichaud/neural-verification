"""
This script loads up the results of each run with a different l1, for each
task, and selects the network with the right l1 for each task. It 
creates a directory in the chosen directory for that tasks and copies
the results of the run with the right l1 into that directory.
"""

import os
import sys
import subprocess

from tqdm.auto import tqdm

import numpy as np
import torch

l1s = [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

tasks = [
    "rnn_parity_last2_numerical",
    "rnn_parity_last3_numerical",
    "rnn_parity_last4_numerical",
    "rnn_parity_numerical",
    "rnn_sum_numerical",
    "rnn_sum_last2_numerical",
    "rnn_sum_last3_numerical",
    "rnn_sum_last4_numerical",
    "rnn_sum_last5_numerical",
    "rnn_sum_last6_numerical",
    "rnn_sum_last7_numerical",
    "rnn_identity_numerical",
    "rnn_prev1_numerical",
    "rnn_prev2_numerical",
    "rnn_prev3_numerical",
    "rnn_prev4_numerical",
    "rnn_prev5_numerical",
    "rnn_unique2_numerical",
    "rnn_diff_last2_numerical",
    "rnn_abs_value_of_diff_numerical",
    "rnn_abs_value_numerical",
    "rnn_diff_of_abs_value_numerical",
    "rnn_min_numerical",
    "rnn_max_numerical",
    "rnn_bitwise_not",
    "rnn_bitwise_and",
    "rnn_bitwise_or",
    "rnn_bitwise_xor",
    "bit_addition",
    "rnn_majority0_1_numerical",
    "rnn_majority0_2_numerical",
    "rnn_majority0_3_numerical",
    "rnn_evens_detector_numerical",
    "rnn_evens_counter_numerical",
    "rnn_perfect_square_detector_numerical",
    "rnn_bit_palindromes_numerical",
    "rnn_balanced_parenthesis_numerical",
    "rnn_parity_of_index_numerical",
    "rnn_parity_of_zeros_numerical",
    "rnn_alternating_last3_numerical",
    "rnn_alternating_last4_numerical",
    "rnn_bit_shift_right_numerical",
    "rnn_bit_dot_prod_mod2_numerical",
    "rnn_div_3_numerical",
    "rnn_div_5_numerical",
    "rnn_div_7_numerical",
    "rnn_add_mod_3_numerical",
    "rnn_add_mod_4_numerical",
    "rnn_add_mod_5_numerical",
    "rnn_add_mod_6_numerical",
    "rnn_add_mod_7_numerical",
    "rnn_add_mod_8_numerical",
    "rnn_dithering_numerical",
    "rnn_newton_freebody_numerical",
    "rnn_newton_gravity_numerical",
    "rnn_newton_spring_numerical",
    "rnn_newton_magnetic_numerical",
    "rnn_base_3_addition",
    "rnn_base_4_addition",
    "rnn_base_5_addition",
    "rnn_base_6_addition",
    "rnn_base_7_addition",
    "rnn_base_8_addition"
]

print(len(tasks))


if __name__ == '__main__':
    home_dir = "/om2/user/ericjm/neural-verification"
    save_dir = "/om2/user/ericjm/neural-verification/regularized_models_final"
    backup_dir = "/om2/user/ericjm/neural-verification/first_paper"
    # create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for task in tqdm(tasks):
        # load up the metrics for each run with different l1
        best_accs = dict()
        best_losses = dict()
        for l1 in l1s:
            try:
                metrics = torch.load(os.path.join(home_dir, f"regularized_models_{l1}", task, "regularized_metrics.pt"))
                # the model we care about is `regularized_model_best`. This is the model with the lowest
                # test loss. So compute the index at which the test loss is lowest
                test_losses = metrics['test_losses']
                best_idx = np.argmin(test_losses)
                best_losses[l1] = test_losses[best_idx]
                # what is the accuracy for this model?
                best_acc = metrics['test_accuracies'][best_idx]
                best_accs[l1] = best_acc
            except:
                best_losses[l1] = np.inf
                best_accs[l1] = 0.0
                print(f"Error loading metrics for {task} with l1={l1}")
        # find the largest l1 where the accuracy is still 100%
        best_l1 = None
        for l1 in l1s[::-1]:
            if best_accs[l1] == 1.0:
                best_l1 = l1
                break

        task_dir = os.path.join(save_dir, task)
        
        if best_l1 is None:
            # copy the `first_paper` contents
            subprocess.run(["cp", "-r", os.path.join(backup_dir, task), task_dir])
        else:
            # move down by 2 l1s (10x), if that model's loss is better
            safer_l1 = l1s[max(0, l1s.index(best_l1) - 2)]
            if best_losses[safer_l1] < best_losses[best_l1]:
                choice_l1 = safer_l1
            else:
                choice_l1 = best_l1
            
            # copy the contents of the directory with the choice_l1 into the task's directory
            subprocess.run(["cp", "-r", os.path.join(home_dir, f"regularized_models_{choice_l1}", task), task_dir])
