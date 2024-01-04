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


    ### Two usage options:
    ### python evolution_tree.py <task_name>
    ### python evolution_tree.py <task_name> <simplifier> <model> <options...>

    dataset_path = "../tasks/"
    original_raw_models_path = "./rnn_tests/raw_models/"
    original_processed_models_path = "./rnn_tests/processed_models/"

    task_names_with_models = [folder for folder in os.listdir(original_raw_models_path) if os.path.isdir(os.path.join(original_raw_models_path, folder))]
    task_names_with_create_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    task_names_with_datasets = [folder for folder in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, folder, "data.pt"))]
    with open("../search.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    task_names_in_yaml = list(config.keys())

    task_names = list(set(task_names_with_models) & set(task_names_with_datasets) & set(task_names_in_yaml))
    task_names2 = list(set(task_names_with_models) & set(task_names_with_create_datasets) & set(task_names_in_yaml))
    print(task_names)
    print(set(task_names2)-set(task_names))
    
    for task in task_names:
        raw_models_path = original_raw_models_path + task + "/"
        processed_models_path = original_processed_models_path + task + "/"

        print("Making evolution tree for models in " + processed_models_path + ".")
        tree = evolution_tree.EvolutionTree(raw_models_path, processed_models_path, task)
        tree_fname = "./rnn_tests/evolution_tree_" + task
        log_fname = "./rnn_tests/evolution_tree_" + task + ".txt"
        if os.path.exists(tree_fname):
            tree.load_tree(tree_fname)
        else:
            tree.init_from_raw_models()

        simplifiers_in_order = tuple(map(lambda s: "normalizers/" + s + ".py", ("prune_dead_neurons_rnn", "rotate_hidden_space", "prune_hidden_dim", "align_hidden_space", "quantize_weights")))
        model = processed_models_path + task + "/model_perfect.pt"

        short_names_to_simplifiers = {val:key for key, val in evolution_tree.simplifier_short_names.items()}
        normalizer_tree = ["prune",
                ["jnf",
                    ["compress",
                        ["quantize",
                            ["prune"]]]],
                ["rotate",
                    ["compress",
                        ["quantize",
                            ["prune"]]]],
                ["align",
                    ["compress",
                        ["quantize",
                            ["prune"]]]],
                ["diagonalize",
                    ["compress",
                        ["quantize",
                            ["prune"]]]],
        ]
        progress_bar = tqdm.tqdm(total=17, desc="Simplifying", unit="iteration")

        def normalize(normalizer_tree, old_model):
            simplifier = short_names_to_simplifiers[normalizer_tree[0]]
            new_model = old_model[:-3] + "_" + "_".join([evolution_tree.simplifier_short_names[simplifier]] + list(map(str, evolution_tree.default_parameters[simplifier]))) + ".pt"
            try:
                tree.simplify(old_model, new_model, simplifier, tuple(map(str, evolution_tree.default_parameters[simplifier])))
            except ValueError as e:
                print(old_model, simplifier, evolution_tree.default_parameters[simplifier])
                raise e
            tree.save_tree(tree_fname)
            tree.log_tree(log_fname)
            progress_bar.update(1)
            for subtree in normalizer_tree[1:]:
                normalize(subtree, new_model)
        normalize(normalizer_tree, model)
        progress_bar.close()
