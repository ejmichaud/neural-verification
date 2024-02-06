import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import os
import itertools

from neural_verification import MLP, MLPConfig
from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
)

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
print(no_create_dataset)
print(no_dataset)
print(no_model)
if no_create_dataset:
    raise ValueError
if no_dataset:
    raise ValueError
if no_model:
    raise ValueError


for task_name in task_names:

    task_path = original_processed_models_path + task_name + "/model_perfect_whiten_0.1_jnf_0.7_toeplitz_debias_0.1_quantize_0.01.pt"
    if not os.path.exists(task_path):
        raise ValueError("trained network not found: " + task_path)

    if torch.cuda.is_available():
        weights = torch.load(task_path)
    else:
        weights = torch.load(task_path, map_location=torch.device('cpu'))

    def betterprint(X):
        print(goodrepr(X))
    def goodrepr(X):
        if isinstance(X, dict):
            return "{" + ",\n".join([key + ":\n" + goodrepr(val) for key, val in X.items()]) + "\n}"
        if type(X) == list:
            return "[" + "\n".join([goodrepr(x) for x in X]) + "]"
        elif isinstance(X, np.ndarray):
            return np.array2string(X, max_line_width=100, precision=4, suppress_small=True)
        elif torch.is_tensor(X):
            return goodrepr(X.numpy())
        else:
            return str(X)
    #        print(type(X))
    #        raise ValueError

    print(task_name)
#    betterprint(weights)

    task = task_name

    # Read search.yaml
    with open("../search.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    task_config = config[task]

    task_args = task_config['args']
    vectorize_input = task_args['vectorize_input']
    loss_fn = task_args['loss_fn']  # 'cross_entropy' for example

    if torch.cuda.is_available():
        weights = torch.load(task_path)
    else:
        weights = torch.load(task_path, map_location=torch.device('cpu'))


    with torch.no_grad():
        all_int = True
        for name, value in weights.items():
            all_int = all_int and np.all(np.abs(np.round(value.numpy()) - value.numpy()) < 0.3)
        if all_int:
            # Get the RNN shape
            hidden_mlp_depth = 0
            output_mlp_depth = 0
            for key in weights.keys():
                if key[:4] == 'hmlp' and key[-6:] == 'weight':
                    hidden_mlp_depth += 1
                if key[:4] == 'ymlp' and key[-6:] == 'weight':
                    output_mlp_depth += 1
            hidden_dim = weights['hmlp.mlp.' + str(hidden_mlp_depth*2-2) + '.weight'].shape[0]
            output_dim = weights['ymlp.mlp.' + str(output_mlp_depth*2-2) + '.weight'].shape[0]
            input_dim = weights['hmlp.mlp.0.weight'].shape[1] - hidden_dim
            if hidden_mlp_depth >= 2:
                hidden_mlp_width = weights['hmlp.mlp.0.weight'].shape[0]
            else:
                hidden_mlp_width = hidden_dim
            if output_mlp_depth >= 2:
                output_mlp_width = weights['ymlp.mlp.0.weight'].shape[0]
            else:
                output_mlp_width = output_dim

            inputs = ["x" + str(i) for i in range(input_dim)]
            hiddens = ["h" + str(i) for i in range(hidden_dim)]
            outputs = ["y" + str(i) for i in range(output_dim)]
            inputs = hiddens + inputs

            preactivations = inputs
            for depth in range(hidden_mlp_depth):
                if depth != 0:
                    preactivations = [task_args['activation'] + "(" + val + ")" for val in preactivations]
                weight = weights['hmlp.mlp.' + str(depth*2) + '.weight'].numpy()
                bias = weights['hmlp.mlp.' + str(depth*2) + '.bias'].numpy()
                new_preactivations = []
                for i in range(weight.shape[0]):
                    new_expression = ""
                    for j in range(weight.shape[1]):
                        if int(weight[i][j]) != 0:
                            new_expression = new_expression + (" + " if weight[i][j] > 0 else " - ")
                            if abs(int(weight[i][j])) != 1:
                                new_expression = new_expression + str(abs(int(weight[i][j]))) + "*" + preactivations[j]
                            else:
                                new_expression = new_expression + preactivations[j]
                    if int(bias[i]) != 0:
                        new_expression = new_expression + (" + " if bias[i] > 0 else " - ")
                        new_expression = new_expression + str(abs(int(bias[i])))
                    if new_expression[:3] == " + ":
                        new_expression = new_expression[3:]
                    if new_expression[:3] == " - ":
                        new_expression = "-" + new_expression[3:]
                    new_preactivations.append(new_expression)
                preactivations = new_preactivations
            s = ", ".join(hiddens) + " = " + ", ".join(preactivations)

            preactivations = hiddens
            for depth in range(output_mlp_depth):
                if depth != 0:
                    preactivations = [task_args['activation'] + "(" + val + ")" for val in preactivations]
                weight = weights['ymlp.mlp.' + str(depth*2) + '.weight'].numpy()
                bias = weights['ymlp.mlp.' + str(depth*2) + '.bias'].numpy()
                new_preactivations = []
                for i in range(weight.shape[0]):
                    new_expression = ""
                    for j in range(weight.shape[1]):
                        if int(weight[i][j]) != 0:
                            new_expression = new_expression + (" + " if weight[i][j] > 0 else " - ")
                            if abs(int(weight[i][j])) != 1:
                                new_expression = new_expression + str(abs(int(weight[i][j]))) + "*" + preactivations[j]
                            else:
                                new_expression = new_expression + preactivations[j]
                    if int(bias[i]) != 0:
                        new_expression = new_expression + (" + " if bias[i] > 0 else " - ")
                        new_expression = new_expression + str(abs(int(bias[i])))
                    if new_expression[:3] == " + ":
                        new_expression = new_expression[3:]
                    if new_expression[:3] == " - ":
                        new_expression = "-" + new_expression[3:]
                    new_preactivations.append(new_expression)
                preactivations = new_preactivations
            s = s + "\n" + ", ".join(outputs) + " = " + ", ".join(preactivations)

            print(s)
        else:
            betterprint(weights)


    print("")
