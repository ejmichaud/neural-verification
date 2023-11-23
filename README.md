
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

## Development notes
- [] Set torch seed in all `create_dataset.py` scripts and re-run.

## Structure
```
.
├── literature
├── models
├── neural_verification
│   ├── neural_verification.py
│   └── setup.py
├── README.md
├── scripts
|   ├── rnn_architecture_search.py
|   └── rnn_train.py
└── tasks
    ├── palindromes
        ├── config.py
        ├── create_dataset.py
        └── train.py
├── normalizers
│   ├── make_example_MLP.py
│   ├── delete_dead_neurons.py
│   ├── combine_duplicate_neurons.py
    ...
```
For each task, we create a directory under `tasks/` and create scripts for defining the dataset and for training a network. At a minimum, the code in a task's directory should produce train and test datasets for the task and also save the weights of a model which was trained on the train set and achieves good performance on the test set. Do not commit large files to git! Datasets and model weights, unless they are extremely small, should be uploaded to the shared Dropbox folder: https://www.dropbox.com/scl/fo/deeu5ifrpi8vn8hnujlwc/h?rlkey=zpo1ebv08wazrzrxc7rpn9pct&dl=0


## Installation

We've defined several helpful objects (data loaders, tokenizer, and a transformer implementation) inside of the `neural-verification` module, which you can install with:

```
pip install -e neural_verification
```
This requires PyTorch

## Running architecture search
You can perform an architecture search on a task by running the `scripts/rnn_architecture_search.py` script. For example, to run an architecture search on the `rnn_prev2_numerical` task, you could run something like:
```
python scripts/rnn_architecture_search.py --data tasks/rnn_prev2_numerical/data.pt --loss_fn "log" --input_dim 1 --output_dim 1 --seeds-per-run 3 --save_dir tasks/rnn_prev2_numerical/arch --steps 5000
```
There are many choices involved in running this script. Inside `scripts/rnn_architecture_search.py`, we basically run `scripts/rnn_train.py` with many different arguments, so you could take a look at that script to see what arguments are available. The most important arguments are the `--loss_fn` argument for choosing between "mse", "log", and "cross_entropy". The "log" option is the 0.5*log(1+x^2) loss function. There is also a flag `--vectorize_input` which will turn ints into
one-hot vectors before passing them to the network. This shouldn't be used for "numerical" tasks where the input is a 1d
vector, but is useful for tasks where the input is something categorical -- for instance if inputs are letters, you may
want the inputs to be one-hot vectors of length 26. You'll also have to choose the `--input_dim` and `--output_dim` arguments
to match the task. For instance, in the example I just gave where the input is a one-hot vector of length 26, you would
set `--input_dim 26`. The `--seeds-per-run` argument controls how many different seeds to use for each architecture. The
`--steps` argument controls how many steps to train for.

With this script, you really just need to create a dataset of input and output tensors. The script should flexible enough
to handle the input sequences which are just a 1d list of real numbers, but can also handle input sequences which are
a list of higher-dimensional vectors. 
