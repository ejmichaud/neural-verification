This folder contains tools for operating different normalizers which can be used to simplify neural networks. A normalizer tries to reduce a neural network into it's "normal form", a standard form of sorts which all neural networks that use the same algorithm can be reduced to.

Currently, we have only added normalizers that work on MLPs, and all code only works on the multiplication dataset. The `normalize_weights_mlp.py` only works on ReLU networks, which are not Anish's multiplication networks, so the normalization functionality remains untested.


To operate the normalizer system, you run `evolution_tree.py`. It will copy a bunch of models from `raw_models/` into `processed_models/` and then let you run normalizers on them, via the line:
```
python evolution_tree.py normalizers/<normalizer>.py processed_models/<model>.pt <normalizer_arg1> <normalizer_arg2> ...
```
This will generate a newly normalized model stored in `processed_models/`. `evolution_tree.txt` will then be generated to show how the models were derived from each other, and the file `evolution_tree` will store this tree structure for later reuse.


Folder structure:
```
 - normalizers/delete_dead_neurons_mlp.py
 - normalizers/combine_duplicate_neurons_mlp.py
 - normalizers/sort_neurons_mlp.py
 - normalizers/retrain_mlp.py
 - normalizers/L1_retrain_mlp.py
 - normalizers/normalize_weights_mlp.py

 - prototypes/make_example_mlp.py
 - prototypes/make_example_rnn.py
 - prototypes/delete_dead_neurons_rnn.py

 - metrics/loss_mlp.py
 - metrics/neuron_count_mlp.py
 - metrics/sparsity_mlp.py
 - metrics/weight_norm_mlp.py

 - evolution_tree.py
 - evolution_tree
 - evolution_tree.txt

 - raw_models/
 - processed_models/
```

File Descriptions:
 - `delete_dead_neurons_mlp.py` contains code that deletes all neurons that are deemed to be "dead". The rule for whether a neuron is dead is if the norm of the vector of input weights times the norm of the vector of output weights times 1.1 is below some threshold defined by `prune_threshold` at the top of the code. After these neurons are pruned, a bunch of dead neurons with zero input and output weights and zero bias get added to retain the shape.
 - `combine_duplicate_neurons_mlp.py` contains code that combines pairs of neurons that are deemed to "compute the same thing". The rule for whether neurons compute the same thing are if the vectors of their input weights (plus bias) are closer than some threshold defined by `prune_threshold` at the top of the code. epsilon at the top of the code is a threshold for detecting killed/zeroed out neurons. Both neurons are removed, a combined neuron is added at the beginning and a dead neuron is added at the end. This is repeated one neuron at a time until there are no more neurons to combine.
 - `normalize_weights_mlp.py` contains code that normalizes all the rows of all the weight matrices (except the last one). This is only doable for ReLU networks which have the property that aReLU(x) = ReLU(ax) for a>0.
 - `sort_neurons_mlp.py` contains code that sorts all the hidden neurons such that the ones with the highest sum of absolute incoming and outgoing weights comes first.
 - `retrain_mlp.py` contains code that retrains a given neural network for some number of iterations using SGD of some initial learning rate.
 - `L1_retrain_mlp.py` contains code that retrains a given neural network with L1 norm of some strength for some number of iterations using SGD of some initial learning rate.

 - `make_example_mlp.py` contains code that constructs a randomly initialized MLP and saves it in `sample_model.pt`.


Normalizers:
	Exact Normalizers:
		sort neurons (`sort_neurons_mlp.py`)
		scale the input weights and output weights of ReLUs (`normalize_weights_mlp.py`)
	Approximate Normalizers:
		delete dead neurons (`delete_dead_neurons_mlp.py`)
		combine duplicate neurons (`combine_duplicate_neurons_mlp.py`)
		combine ReLUs with large biases (Reduces to low rank matrix and cannot combine??)
		ensure that the model is split into modules
	Data Dependent Normalizers:
		delete neurons that have no activation variance
		find and replace identical modules if one identical module is equivalent to another
	Crude Normalizers:
		BIMT
		Hypernet
		retraining (`retrain_mlp.py`)
		L1 activation penalty
		L1 regularization (`L1_retrain_mlp.py`)
