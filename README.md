
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

## Structure
```
.
├── literature
├── models
├── neural_verification
│   ├── neural_verification.py
│   └── setup.py
├── README.md
└── tasks
    ├── palindromes
        ├── config.py
        ├── create_dataset.py
        └── train.py
    ...
```
For each task, we create a directory under `tasks/` and create scripts for defining the dataset and for training a network. At a minimum, the code in a task's directory should produce train and test datasets for the task and also save the weights of a model which was trained on the train set and achieves good performance on the test set. Do not commit large files to git! Datasets and model weights, unless they are extremely small, should be uploaded to the shared Dropbox folder: https://www.dropbox.com/scl/fo/deeu5ifrpi8vn8hnujlwc/h?rlkey=zpo1ebv08wazrzrxc7rpn9pct&dl=0


## Installation

We've defined several helpful objects (data loaders, tokenizer, and a transformer implementation) inside of the `neural-verification` module, which you can install with:

```
pip install -e neural_verification
```
This requires PyTorch


