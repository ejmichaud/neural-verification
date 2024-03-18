
```
                                                             _---~~(~~-_.
                                                     +------{        )   )
   ._________________.                               |   ,   ) -~~- ( ,-' )_
   |.---------------.|<------------------------------+  (  `-,_..`., )-- '_,)
   ||               ||  _ __   ___ _   _ _ __ __ _| |  ( ` _)  (  -~( -_ `,  }
   ||   -._ .-.     || | '_ \ / _ \ | | | '__/ _` | |  (_-  _  ~_-~~~~`,  ,' )
   ||   -._| | |    || | | | |  __/ |_| | | | (_| | |    `~ -^(    __;-,((()))
   ||   -._|"|"|    || |_| |_|\___|\__,_|_|  \__,_|_|          ~~~~ {_ -_(())---+
   ||   -._|.-.|    ||                                                `\  }     |
   ||_______________||                                                  { }     |
   /.-.-.-.-.-.-.-.-.\      __   _____ _ __(_)/__(_) ___ __ _|_|_(_) ___  _ __  |
  /.-.-.-.-.-.-.-.-.-.\     \ \ / / _ \ '__| | |_| |/ __/ _` | __| |/ _ \| '_ \ |
 /.-.-.-.-.-.-.-.-.-.-.\     \ V /  __/ |  | |  _| | (_| (_| | |_| | (_) | | | ||
/______/__________\___o_\     \_/ \___|_|  |_|_| |_|\___\__,_|\__|_|\___/|_| |_||
\_______________________/<------------------------------------------------------+
```

## Summary and State

This is the repository for the preprint "Opening the AI black box: program synthesis via mechanistic interpretability" (https://arxiv.org/abs/2402.05110). Some parts of our pipeline haven't been added yet to the repository. If you'd like to get these asap, please email me at `ericjm` [at] `mit.edu`. 

In this repository currently are the parts of the pipeline needed to create datasets for our tasks and then perform an architecture search on each task to find the smallest RNN capable of solving each one.

## Creating task datasets

Inside the `tasks` directory are a variety of folders each containing a `create_dataset.py` script. To create the dataset for a task, you can just run this script in the corresponding folder. This will create a `data.pt` file in the task's folder with 1e6 sample sequences, split in to train and test sets.

## Installing the neural-verification module
We define several helpful objects (the RNN model, data loaders, etc.) inside of the `neural-verification` module, which you can install with:
```
pip install -e neural_verification
```
This requires PyTorch.

## Training RNNs (architecture search)

To perform an architecture search across a set of tasks, one can run the `scripts/rnn_architecture_search.py` script like so:
```
python scripts/rnn_architecture_search.py --config first_paper_final.yaml --tasks_dir tasks/ --architecture_search_script scripts/rnn_architecture_search.py --save_dir arch_search
```

The `--config` argument specifies a yaml file containing the args to pass to the `rnn_architecture_search.py` script for each task. It will run the architecture search on each task with a separate slurm job.


## Hammering RNNs

The `normalizers` folder contains tools for operating different normalizers which can be used to simplify neural networks. A normalizer tries to reduce a neural network into it's "normal form", a standard form of sorts which all neural networks that use the same algorithm can be reduced to.

Currently, we have normalizers that work on RNNs from `neural_verification/neural_verification.py`.

To operate the normalizer system:
1. `cd` to the `normalizers` directory.
2. Modify and use transfer.py to copy all the `model_perfect.pt` from a folder of task folders to `rnn_tests/raw_models/`.
3. Run `automatic_evolution.py` in order to run the normalizer sequence: [whitening, Jordan normal form, Toeplitz, de-biasing, quantization] on all the copied models.
4. Modify and use transfer.py to copy all the hammered models from `rnn_tests/raw_models/` to some destination folder.

A lot of tools can be used along the way to look into the above process:
1. To use a single normalizer on a single model, you run `evolution_tree.py`, via the line: `python evolution_tree.py normalizers/<normalizer>.py processed_models/<model>.pt <normalizer_arg1> <normalizer_arg2> ...`
`rnn_tests/evolution_tree_*` are all files which store and display all of the models that have been in the normalization system so far, and all measurements of the models.
2. We can view the model weights using `inspect_model.py`, which is easily modifiable for us to perform and print the results of basic computations using the weights.
3. We can view all the weights of all the models using `print_model_weights.py`.
4. We can summarize the overall effect of all the normalizers averaged over the models, using `summary_statistics.py`.

Folder structure:
```
normalizers
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

## MIPS: Converting trained NNs into python code
### Setup
1. cd into the folder ./regression.
2. Download the folder called "first_paper" from dropbpx: https://www.dropbox.com/scl/fo/fqhm7v8euiduchm4eke8b/h?rlkey=pp99bo9v1cgth6os51ro1ejcq&dl=0 which contains already trained models.
3. Raname the folder from "first_paper" to "tasks".
4. Open run.ipynb, click and run! Notice in the ./programs folder, txt files are generated which contain the MIPS-generated python programs. And the csv file ./results.csv is a table summarizing the task, success or failure, and (if success) which regression method nailed the task.

That's it! In case you want to dig into details about how MIPS is implemented, here is some brief summary of these methods and where to find these files.
#### Extraction of variables with integer autoencoder
We find that integer lattices are prevalent in RNN hidden space, for almost all tasks we reporeted in the paper. So we come up with integer autoencoders to convert numerical hidden states into integer lattices. We implment two methods (linear autoencoder and GCD autoencoder) in ./regression/integer_autoencoder.py. In ./regression/auto_encode_RNN.py, we run linear autoencoder and GCD autoencoders with different hyperparameters and select the best integer lattice in terms of minimal description length.

#### Regression methods
After the integer lattice is found, we try three regression methods:
1. Linear regression, implemented in ./regression/linear_regression.ipynb
2. Boolean regression, implemented in ./regression/boolean_regression.ipynb
3. Symbolic regression, implemented in ./regression/symbolic_regression.ipynb and ./regression/symbolic_regression

## Comparison to GPT-4
### Set-Up
To run the GPT baseline on your dataset, you will first need to input your OpenAI token into `/gpt4/prompt.py`. Enter your API key here:

```python
api_key = "your-key-here"
```

#### Running the baseline for a single task

To run the baseline for a single task, use the following command:

```sh
python prompt.py --task <task_name> --iterations <number_of_iterations>
```

Replace `<task_name>` with one of the tasks listed in `/tasks/`. The `<number_of_iterations>` parameter repeats the run for a fixed number, increasing the chance of successfully discovering an algorithm. The default value for iterations is 3.

#### Running the baseline for all tasks

To automatically prompt GPT for all tasks listed at the start of the file in `/gpt4/prompt_all.py`, use the following command:

```sh
python prompt_all.py --iterations <number_of_iterations>
```

Here, `<number_of_iterations>` specifies how many times each task should be repeated.

#### Extracting results and creating a table

To extract the results and generate a `tasks_results.csv` table, run the following command:

```sh
python run_extracted_code.py
```

This command will save all the results in the `task_results.csv` file.

## A note on naming conventions
We changed the names of the tasks for the paper, and we still need to rename the tasks in this repository to reflect that.

