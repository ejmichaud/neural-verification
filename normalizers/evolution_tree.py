import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import subprocess

from neural_verification import MLP


class EvolutionTree():
    """
    This class represents a forest structure whereby the roots of each tree are base models (which are processed from raw models), nodes represent intermediate models, and the edges represent different normalizers that have been applied to the models to produce more normal models.

    All the raw models can be found in one folder raw_models_path, and they each map to models in processed_models_path, where the files have the same name. There are additional models in processed_models_path, and they come in a tree structure. model_locations tells you which paths to take to get to any given model in the tree, while model_tree tells you what paths are available at any given model in the tree. model_parents tells you which model is the parent of any given model.
    """
    def __init__(self, raw_models_path, processed_models_path):
        self.raw_models_path = raw_models_path
        self.processed_models_path = processed_models_path
        self.children = dict()
        self.parents = dict()
        self.roots = set()
        self.nodes = set()
    
    def init_from_raw_models(self):
        file_list = os.listdir(self.raw_models_path)
        for file_name in file_list:
            raw_model_path = os.path.join(self.raw_models_path, file_name)
            processed_model_path = os.path.join(self.processed_models_path, file_name)

            # Get the weights and biases
            original_weights = torch.load(raw_model_path, map_location=torch.device('cpu'))
            if tuple(sorted(original_weights.keys())) == tuple(sorted(['mlp.0.weight', 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias'])):
                key1_key2_pairs = [('mlp.0.weight', 'linears.0.weight'), ('mlp.0.bias', 'linears.0.bias'), ('mlp.2.weight', 'linears.1.weight'), ('mlp.2.bias', 'linears.1.bias')]
                original_weights = {key2:original_weights[key1] for (key1, key2) in key1_key2_pairs}
            elif tuple(sorted(original_weights.keys())) == tuple(sorted(['linears.' + str(i) + '.weights' for i in range(int(len(original_weights)//2))] + ['linears.' + str(i) + '.bias' for i in range(int(len(original_weights)//2))])):
                pass
            else:
                print(original_weights.keys())
                raise ValueError
            prefix = 'linears.'
            original_shape = [original_weights[prefix + '0.bias'].shape[0]] + [original_weights[prefix + str(i) + '.bias'].shape[0] for i in range(int(len(original_weights)//2))]
            weights = [original_weights[prefix + str(i) + '.weight'].numpy() for i in range(int(len(original_weights)//2))]
            biases = [original_weights[prefix + str(i) + '.bias'].numpy() for i in range(int(len(original_weights)//2))]

            # Make a model in a standard way with them
            shp = [weights[0].shape[1]] + [bias.shape[0] for bias in biases]
            depth = len(shp)-1
            width = max(shp[1:-1])
            in_dim = shp[0]
            out_dim = shp[-1]
            model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth)
            linear_list = []
            for i in range(depth):
                linear_list.append(nn.Linear(shp[i], shp[i+1]))
            model.linears = nn.ModuleList(linear_list)
            model.shp = shp
            model.load_state_dict(original_weights)

            # Save the model
            torch.save(model.state_dict(), processed_model_path)

            self.roots.add(file_name)
            if file_name not in self.children:
                self.children[file_name] = set()
            self.parents[file_name] = None
            self.nodes.add(file_name)

    def save_tree(self, tree_fname):
        with open(tree_fname, 'wb') as f:
            pickle.dump([self.roots, self.nodes, self.children, self.parents], f)

    def load_tree(self, tree_fname):
        with open(tree_fname, 'rb') as f:
            self.roots, self.nodes, self.children, self.parents = pickle.load(f)

    def infer_tree(self):
        file_list = os.listdir(self.processed_models_path)
        file_list = sorted(file_list, key=lambda x: (len(x), x))  # sort by length first and then alphabetically
        self.children = dict()
        self.parents = dict()
        self.roots = set()
        self.nodes_list = []
        for file_name in file_list:
            self.nodes_list.append(file_name)
            self.children[file_name] = set()
            is_root = True
            for potential_parent in reversed(self.nodes_list):
                if file_name.startswith(potential_parent):
                    self.parents[file_name] = potential_parent
                    self.children[potential_parent].add(file_name)
                    is_root = False
                    break
            if is_root:
                self.parents[file_name] = None
                self.roots.add(file_name)
        self.nodes = set(self.nodes_list)

    def log_tree(self, log_fname):
        s = ""
        def log_tree_helper(prefix, struct, prefix_to_cut):
            if prefix_to_cut == 0:
                s = prefix + struct[prefix_to_cut:] + "\n"
            else:
                s = prefix + struct[prefix_to_cut:-3] + "\n"
            for child in sorted(self.children[struct]):
                s = s + log_tree_helper(prefix + "\t", child, len(struct)-2)
            return s

        for root in self.roots:
            s = s + log_tree_helper("", root, 0)
        with open(log_fname, 'w') as f:
            f.write(s)

    def simplify(self, model, new_model, simplifier, other_options):  # other_options is a list full of arguments to the argparsed python files which perform model simplification, this list is in the order defined by the simplifiers_information variable.
        arguments = ["python", simplifier, processed_models_path + model[model.find("/")+1:], processed_models_path + new_model[new_model.find("/")+1:]] + sum([[option_name, option_value] for option_name, option_value in zip(simplifiers_information[simplifier], other_options)], [])
        print(arguments)
        subprocess.run(arguments)
        shortened_model = model[model.find("/")+1:]
        shortened_new_model = new_model[new_model.find("/")+1:]
        self.children[shortened_model].add(shortened_new_model)
        self.children[shortened_new_model] = set()
        self.parents[shortened_new_model] = shortened_model
        self.nodes.add(shortened_new_model)


simplifiers_information = {
        "normalizers/combine_duplicate_neurons_mlp.py": ("-t", "-e"),
        "normalizers/delete_dead_neurons_mlp.py": ("-t",),
        "normalizers/normalize_weights_mlp.py": (),
        "normalizers/retrain_mlp.py": ("-n", "-l"),
        "normalizers/sort_neurons_mlp.py": (),
}
simplifier_short_names = {
        "normalizers/combine_duplicate_neurons_mlp.py": "deduplicate",
        "normalizers/delete_dead_neurons_mlp.py": "prune",
        "normalizers/normalize_weights_mlp.py": "normalize",
        "normalizers/retrain_mlp.py": "retrain",
        "normalizers/sort_neurons_mlp.py": "sort",
}


raw_models_path = "./raw_models/"
processed_models_path = "./processed_models/"
print("Making evolution tree for models in " + processed_models_path + ".")
evolution_tree = EvolutionTree(raw_models_path, processed_models_path)
tree_fname = "evolution_tree"
log_fname = "evolution_tree.txt"
if os.path.exists(tree_fname):
    print("Loading evolution tree from " + tree_fname + ".")
    evolution_tree.load_tree(tree_fname)
else:
    print("Constructing new evolution tree from models in " + raw_models_path + ".")
    evolution_tree.init_from_raw_models()

if len(sys.argv) == 2:
    other_args = sys.argv[1:]
    simplifier = sys.argv[1]
    raise ValueError("User requests to use the " + simplifier + " simplifier, but must also request a model.")
elif len(sys.argv) >= 3:
    other_args = sys.argv[1:]
    simplifier = sys.argv[1]
    model = sys.argv[2]
    print("User requests to use the " + simplifier + " simplifier on the " + model + " model.")
    other_options = sys.argv[3:]
    new_model = model[:-3] + "_" + "_".join([simplifier_short_names[simplifier]] + other_options) + ".pt"
    if simplifier not in simplifiers_information:
        raise ValueError("Not a valid simplifier name: " + simplifier_args_and_dict["name"])
    elif len(other_options) != len(simplifiers_information[simplifier]):
        helper_string = subprocess.run(["python", simplifier, "--help"])
        raise ValueError("Invalid number of arguments for using the chosen simplifier model. Help information:\n" + helper_string)
    evolution_tree.simplify(model, new_model, simplifier, other_options)
else:
    print("User did not choose to use a simplifier.")


print("Saving evolution tree in " + tree_fname + ".")
evolution_tree.save_tree(tree_fname)
print("Drawing evolution tree in " + log_fname + ".")
evolution_tree.log_tree(log_fname)
