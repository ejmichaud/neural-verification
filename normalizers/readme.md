This folder contains various different normalizers which can be used to simplify neural networks. A normalizer tries to reduce a neural network into it's "normal form", a standard form of sorts which all neural networks that use the same algorithm can be reduced to.

Currently, we have only added normalizers that work on MLPs.

```
Files:
 - make_example_mlp.py
 - delete_dead_neurons.py
 - combine_duplicate_neurons.py

File Descriptions:
 - make_example_mlp.py contains code that constructs a randomly initialized MLP and saves it in sample_model.pt.
 - delete_dead_neurons.py contains code that deletes all neurons that are deemed to be ``dead''. The rule for whether a neuron is dead is if the norm of the vector of input weights times the norm of the vector of output weights times 1.1 is below some threshold defined by prune_threshold at the top of the code. After these neurons are pruned, a bunch of dead neurons with zero input and output weights and zero bias get added to retain the shape.
 - combine_duplicate_neurons.py contains code that combines pairs of neurons that are deemed to ``compute the same thing''. The rule for whether neurons compute the same thing are if the vectors of their input weights (plus bias) are closer than some threshold defined by prune_threshold at the top of the code. epsilon at the top of the code is a threshold for detecting killed/zeroed out neurons. Both neurons are removed, a combined neuron is added at the beginning and a dead neuron is added at the end. This is repeated one neuron at a time until there are no more neurons to combine.

```
