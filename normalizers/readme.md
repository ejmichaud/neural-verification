This folder contains tools for operating different normalizers which can be used to simplify neural networks. A normalizer tries to reduce a neural network into it's "normal form", a standard form of sorts which all neural networks that use the same algorithm can be reduced to.

Currently, we have normalizers that work on RNNs from `neural_verification/neural_verification.py`.

To operate the normalizer system:
1. Modify and use transfer.py to copy all the `model_perfect.pt` from a folder of task folders to `rnn_tests/raw_models/`.
2. Run `automatic_evolution.py` in order to run the normalizer sequence: [whitening, Jordan normal form, Toeplitz, de-biasing, quantization] on all the copied models.
3. Modify and use transfer.py to copy all the hammered models from `rnn_tests/raw_models/` to some destination folder.

A lot of tools can be used along the way to look into the above process:
1. To use a single normalizer on a single model, you run `evolution_tree.py`, via the line: `python evolution_tree.py normalizers/<normalizer>.py processed_models/<model>.pt <normalizer_arg1> <normalizer_arg2> ...`
`rnn_tests/evolution_tree_*` are all files which store and display all of the models that have been in the normalization system so far, and all measurements of the models.
2. We can view the model weights using `inspect_model.py`, which is easily modifiable for us to perform and print the results of basic computations using the weights.
3. We can view all the weights of all the models using `print_model_weights.py`.
4. We can summarize the overall effect of all the normalizers averaged over the models, using `summary_statistics.py`.

Folder structure:
```
.
├── normalizers
|   ├── debias.py
|   ├── jnf_transition.py
|   ├── quantize_weights.py
|   ├── toeplitz.py
|   └── whitening.py
├── metrics
│   └── all_rnn.py
├── rnn_tests
│   ├── hammered_models
│   ├── processed_models
│   └── raw_models
│       ├── task1
│       │   └── model_perfect.pt
│       ├── task2
│       │   └── model_perfect.pt
│       ...
├── automatic_evolution.py
├── evolution_tree.py
├── inspect_model.py
├── print_all_weights.py
├── readme.md
├── summary_statistics.py
└── transfer.py
```
