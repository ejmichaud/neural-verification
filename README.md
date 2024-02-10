
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
TODO

## Extraction of variables with integer autoencoder
TODO

## Symbolic regression
TODO

## Comparison to GPT-4
TODO

## A note on naming conventions
We changed the names of the tasks for the paper, and we still need to rename the tasks in this repository to reflect that.

