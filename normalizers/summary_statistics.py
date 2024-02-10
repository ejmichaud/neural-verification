import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import subprocess
import tqdm
import yaml

import evolution_tree

if __name__ == "__main__":

    dataset_path = "../tasks/"
    original_raw_models_path = "./rnn_tests/raw_models/"
    original_processed_models_path = "./rnn_tests/processed_models/"

    task_names_with_models = [folder for folder in os.listdir(original_raw_models_path) if os.path.isdir(os.path.join(original_raw_models_path, folder))]
    task_names_with_create_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    task_names_with_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, folder, "data.pt"))]
    with open("../search.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    task_names_in_yaml = list(config.keys())

    task_names = set(task_names_in_yaml)
    no_create_dataset = set(task_names) - set(task_names_with_create_datasets)
    no_dataset = set(task_names) - set(task_names_with_datasets)
    no_model = set(task_names_in_yaml) - set(task_names_with_models)
    if no_create_dataset:
        raise ValueError
    if no_dataset:
        raise ValueError
    if no_model:
        raise ValueError

    task_names_with_trained_models = task_names

   
    normalize_sequences = [
            ["whiten", "jnf", "toeplitz", "debias", "quantize"],
    ]

    for normalize_sequence in normalize_sequences:
        print("Analyzing results for evolution sequence: " + ", ".join(normalize_sequence))
        progress_bar = tqdm.tqdm(total=len(task_names_with_trained_models), desc="Evaluating", unit="iteration")
        original_results = dict()
        evolved_results = dict()
        for task in task_names_with_trained_models:
            raw_models_path = original_raw_models_path + task + "/"
            processed_models_path = original_processed_models_path + task + "/"

            tree = evolution_tree.EvolutionTree(raw_models_path, processed_models_path, task)
            tree_fname = "./rnn_tests/evolution_tree_" + task
            log_fname = "./rnn_tests/evolution_tree_" + task + ".txt"
            if os.path.exists(tree_fname):
                tree.load_tree(tree_fname)
            else:
                raise ValueError("Evolution tree not found")

            model = "model_perfect.pt"

            short_names_to_simplifiers = {val:key for key, val in evolution_tree.simplifier_short_names.items()}
            
            evolved_model = model[:-3] + "".join(["".join(["_" + name] + list(map(lambda s: "_" + str(s), evolution_tree.default_parameters[short_names_to_simplifiers[name]]))) for name in normalize_sequence]) + ".pt"

            original_results[task] = tree.evaluate_metrics(model)
            evolved_results[task] = tree.evaluate_metrics(evolved_model)
            progress_bar.update(1)
        progress_bar.close()

        metrics = list(evolved_results[list(original_results.keys())[0]].keys())
        original_means = [0 for i in range(len(metrics))]
        evolved_means = [0 for i in range(len(metrics))]
        for i, metric in enumerate(metrics):
            for task in task_names_with_trained_models:
                original_means[i] += original_results[task][metric] / len(original_results)
                evolved_means[i] += evolved_results[task][metric] / len(original_results)
        print("\n".join(["Before, average " + metric + ": " + str(result) for metric, result in zip(metrics, original_means)]))
        print("\n".join(["After, average " + metric + ": " + str(result) for metric, result in zip(metrics, evolved_means)]))

        longest_name = max(task_names_with_trained_models, key=len)
        for task in sorted(task_names_with_trained_models):
            s = task + " "*(len(longest_name)-len(task))
            for metric in ("weights", "biases", "params", "norm", "accuracy", "int_weights", "int_biases"):
                longest_metric_value = max([len(str(evolved_results[task2][metric])) for task2 in sorted(task_names_with_trained_models)])
                s += "  " + metric + " " + str(evolved_results[task][metric]) + " "*(longest_metric_value-len(str(evolved_results[task][metric])))
            print(s)
        print("")
