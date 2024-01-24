import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import subprocess


class EvolutionTree():
    """
    This class represents a forest structure whereby the roots of each tree are base models (which are processed from raw models), nodes represent intermediate models, and the edges represent different normalizers that have been applied to the models to produce more normal models.

    All the raw models can be found in one folder raw_models_path, and they each map to models in processed_models_path, where the files have the same name. There are additional models in processed_models_path, and they come in a tree structure. model_locations tells you which paths to take to get to any given model in the tree, while model_tree tells you what paths are available at any given model in the tree. model_parents tells you which model is the parent of any given model.
    """
    def __init__(self, raw_models_path, processed_models_path, task):
        self.raw_models_path = raw_models_path
        self.processed_models_path = processed_models_path
        os.makedirs(self.processed_models_path, exist_ok=True)
        self.task = task
        self.children = dict()
        self.parents = dict()
        self.roots = set()
        self.nodes = set()
        self.metrics = dict()
    
    def init_from_raw_models(self):
        file_list = os.listdir(self.raw_models_path)
        for file_name in file_list:
            raw_model_path = os.path.join(self.raw_models_path, file_name)
            processed_model_path = os.path.join(self.processed_models_path, file_name)

            # Get the weights and biases
            weights = torch.load(raw_model_path, map_location=torch.device('cpu'))

            # Save the model
            torch.save(weights, processed_model_path)

            self.roots.add(file_name)
            if file_name not in self.children:
                self.children[file_name] = set()
            self.parents[file_name] = None
            self.nodes.add(file_name)
            self.metrics[file_name] = self.evaluate_metrics(file_name)

    def save_tree(self, tree_fname):
        with open(tree_fname, 'wb') as f:
            pickle.dump([self.roots, self.nodes, self.children, self.parents, self.metrics], f)

    def load_tree(self, tree_fname):
        with open(tree_fname, 'rb') as f:
            self.roots, self.nodes, self.children, self.parents, self.metrics = pickle.load(f)

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
        self.metrics = {node: self.evaluate_metrics(node) for node in self.nodes}

    def log_tree(self, log_fname):
        s = ""
        def log_tree_helper(prefix, struct, prefix_to_cut):
            if prefix_to_cut == 0:
                s = prefix + struct[prefix_to_cut:]
            else:
                s = prefix + struct[prefix_to_cut:-3]
            s = s + " "*(60-len(s)) + "".join([metric_name + ":" + str(metric_value) + " "*(24-len(str(metric_value))) for metric_name, metric_value in self.metrics[struct].items()]) + "\n"
            for child in sorted(self.children[struct]):
                s = s + log_tree_helper(prefix + "  ", child, len(struct)-2)
            return s

        for root in self.roots:
            s = s + log_tree_helper("", root, 0)
        with open(log_fname, 'w') as f:
            f.write(s)

    def simplify(self, model, new_model, simplifier, other_options):  # other_options is a list full of arguments to the argparsed python files which perform model simplification, this list is in the order defined by the simplifiers_information variable.
        arguments = ["python", simplifier, self.task, self.processed_models_path + model[model.rfind("/")+1:], self.processed_models_path + new_model[new_model.rfind("/")+1:]] + sum([[option_name, option_value] for option_name, option_value in zip(simplifiers_information[simplifier], other_options)], [])
        result = subprocess.run(arguments, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr
#        print(stdout)
        if stderr:
            print(arguments)
            print(stdout)
            raise ValueError(stderr)
        shortened_model = model[model.rfind("/")+1:]
        shortened_new_model = new_model[new_model.rfind("/")+1:]
        self.children[shortened_model].add(shortened_new_model)
        self.children[shortened_new_model] = set()
        self.parents[shortened_new_model] = shortened_model
        self.nodes.add(shortened_new_model)
        self.metrics[shortened_new_model] = self.evaluate_metrics(shortened_new_model)

    def evaluate_metrics(self, model):
        results = {}
        for metric_file, output_entries in metric_names.items():
            arguments = ["python", metric_file, self.task, self.processed_models_path + model]
            result = subprocess.run(arguments, capture_output=True, text=True)
            stdout = result.stdout
            stderr = result.stderr
            if stderr:
                print(arguments)
                print(stdout)
                raise ValueError(stderr)
            for metric_name, search_word in output_entries:
                start_position = stdout.find(search_word)+len(search_word)
                end_position = stdout.find("\n", start_position)
                metric_value = float(stdout[start_position:end_position])
                results[metric_name] = metric_value
        return results
            

simplifiers_information = {
        "normalizers/prune_dead_neurons_rnn.py": ("-t",),
        "normalizers/rotate_hidden_space.py": ("-t",),
        "normalizers/align_hidden_space.py": ("-t",),
        "normalizers/diagonalize_transition.py": (),
        "normalizers/jnf_transition.py": (),
        "normalizers/jnf_transition2.py": ("-e",),
        "normalizers/toeplitz.py": (),
        "normalizers/minimize_description_length.py": ("-t",),
        "normalizers/rescale_input.py": (),
        "normalizers/rescale_output.py": (),
        "normalizers/prune_hidden_dim.py": ("-t",),
        "normalizers/quantize_weights.py": ("-t",),
        "normalizers/whitening.py": ("-e",),
        "normalizers/debias.py": ("-e",),
        "normalizers/noread.py": ("-e",),
        "normalizers/activation_pruner.py": ("-e",),
        "normalizers/rescale_relu.py": (),
}
default_parameters = {
        "normalizers/prune_dead_neurons_rnn.py": (0.1,),
        "normalizers/rotate_hidden_space.py": (500,),
        "normalizers/align_hidden_space.py": (500,),
        "normalizers/diagonalize_transition.py": (),
        "normalizers/jnf_transition.py": (),
        "normalizers/jnf_transition2.py": (0.7,),
        "normalizers/toeplitz.py": (),
        "normalizers/minimize_description_length.py": (800,),
        "normalizers/rescale_input.py": (),
        "normalizers/rescale_output.py": (),
        "normalizers/prune_hidden_dim.py": (0.1,),
        "normalizers/quantize_weights.py": (0.01,),
        "normalizers/whitening.py": (0.1,),
        "normalizers/debias.py": (0.1,),
        "normalizers/noread.py": (0.1,),
        "normalizers/activation_pruner.py": (0.01,),
        "normalizers/rescale_relu.py": (),
}
simplifier_short_names = {
        "normalizers/prune_dead_neurons_rnn.py": "prune",
        "normalizers/rotate_hidden_space.py": "rotate",
        "normalizers/align_hidden_space.py": "align",
        "normalizers/diagonalize_transition.py": "diagonalize",
        "normalizers/jnf_transition.py": "jnf",
        "normalizers/jnf_transition2.py": "jnf2",
        "normalizers/toeplitz.py": "toeplitz",
        "normalizers/minimize_description_length.py": "mdl",
        "normalizers/rescale_input.py": "inrescale",
        "normalizers/rescale_output.py": "outrescale",
        "normalizers/prune_hidden_dim.py": "compress",
        "normalizers/quantize_weights.py": "quantize",
        "normalizers/whitening.py": "whiten",
        "normalizers/debias.py": "debias",
        "normalizers/noread.py": "noread",
        "normalizers/activation_pruner.py": "actreduce",
        "normalizers/rescale_relu.py": "reluscale",
}
metric_names = {
        "metrics/all_rnn.py": (
            ("neurons", "Neurons: "),
            ("weights", "Weights: "),
            ("biases", "Biases: "),
            ("norm", "Weight norm: "),
            ("hidden_dim", "Hidden dim: "),
            ("loss", "Loss: "),
            ("accuracy", "Accuracy: "),
            ("ac_sparsity", "Activation sparsity: "),
            ("int_weights", "Integer weights: "),
            ("int_biases", "Integer biases: "),
            ("params", "Parameters: "),
        ),
}

if __name__ == "__main__":

    ### Two usage options:
    ### python evolution_tree.py <task_name>
    ### python evolution_tree.py <task_name> <simplifier> <model> <options...>

    raw_models_path = "./rnn_tests/raw_models/"
    processed_models_path = "./rnn_tests/processed_models/"
    task = sys.argv[1]
    print(raw_models_path)
    print(task)
    assert os.path.exists(raw_models_path + task + "/")
    raw_models_path = raw_models_path + task + "/"
    processed_models_path = processed_models_path + task + "/"

    print("Making evolution tree for models in " + processed_models_path + ".")
    evolution_tree = EvolutionTree(raw_models_path, processed_models_path, task)
    tree_fname = "./rnn_tests/evolution_tree_" + task
    log_fname = "./rnn_tests/evolution_tree_" + task + ".txt"
    if os.path.exists(tree_fname):
        print("Loading evolution tree from " + tree_fname + ".")
        evolution_tree.load_tree(tree_fname)
    else:
        print("Constructing new evolution tree from models in " + raw_models_path + ".")
        evolution_tree.init_from_raw_models()

    if len(sys.argv) == 3:
        simplifier = sys.argv[2]
        raise ValueError("User requests to use the " + simplifier + " simplifier, but must also request a model.")
    elif len(sys.argv) >= 4:
        simplifier = sys.argv[2]
        model = sys.argv[3]
        print("User requests to use the " + simplifier + " simplifier on the " + model + " model.")
        other_options = sys.argv[4:]
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
