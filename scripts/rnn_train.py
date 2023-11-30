
"""
This script trains a GeneralRNN on a given task. This script is pretty general.
It can be used with arbitrary combinations of: whether the input is a vector of 
real values or should be a one-hot vector, whether the output is treated
like a regression or classification problem, a vector of real values with 
mse/log loss or a vector of real values (logits) with cross-entropy loss.

For a typical regression problem with 1d real valued input and 1d real valued
output, use either --loss "mse" or --loss "log" and do not use --vectorize_input.
Note that the inputs and outputs can still be integers here, they are just treated
as real values.

For a typical classification problem with 1d integer input and 1d integer output,
use --loss "cross_entropy" and --vectorize_input.

If inputs are lists of vectors rather than lists of integers (such as in binary 
logical operators case where at each sequence position we have a 2d vector),
then do NOT use --vectorize_input, just specify the --input_dim as the dimension
of the vectors (e.g. 2). 
"""

import os
import random
import argparse
import json
import yaml

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from neural_verification import (
    GeneralRNNConfig,
    GeneralRNN,
    cycle,
    FastTensorDataLoader
)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def prepare_xy(x, y, vectorize_input, input_dim, dtype, loss_fn):
    """Reshapes, changes dtype, or vectorizes x and y as needed

    Some scenarios:
        1d inputs and 1d outputs regression (mse or log loss, no vectorization)
            Converts input from (batch_size, seq_len) to (batch_size, seq_len, 1)
            Converts output from (batch_size, seq_len) to (batch_size, seq_len, 1)
            Converts both to dtype
        n-d inputs and m-d outputs regression (mse or log loss, no vectorization)
            Keeps input as (batch_size, seq_len, input_dim)
            Keeps output as (batch_size, seq_len, output_dim)
            Converts both to dtype
        integer inputs which need to be vectorized and cross_entropy loss
            Converts input from (batch_size, seq_len) to (batch_size, seq_len, input_dim)
            Converts input dtype to dtype
            Converts output dtype to int64
        n-d inputs and cross-entropy loss
            Keeps input as (batch_size, seq_len, input_dim)
            Converts input dtype to dtype
            Converts output dtype to int64
    """
    # HANDLE X CONVERSIONS
    if vectorize_input:
        x = x.to(torch.int64)
        x = F.one_hot(x, num_classes=input_dim)
    if len(x.shape) == 2: # (batch_size, seq_len)
        x = x.unsqueeze(2) # (batch_size, seq_len, 1)
    x = x.to(dtype)
    # HANDLE Y CONVERSIONS
    if loss_fn == "cross_entropy":
        y = y.to(torch.int64)
    else:
        if len(y.shape) == 2: # (batch_size, seq_len)
            y = y.unsqueeze(2) # (batch_size, seq_len, 1)
        y = y.to(dtype)
    return x, y

def train(args):
    """Trains a GeneralRNN on a given task.
    """
    assert args.loss_fn in ["mse", "log", "cross_entropy"], "loss must be one of 'mse', 'log', 'cross_entropy'"

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(args.seed) # set seed for model initialization
    activation = getattr(torch.nn, args.activation)
    model_config = GeneralRNNConfig(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        hidden_mlp_depth=args.hidden_mlp_depth,
        hidden_mlp_width=args.hidden_mlp_width,
        output_mlp_depth=args.output_mlp_depth,
        output_mlp_width=args.output_mlp_width,
        activation=activation
    )
    model = GeneralRNN(model_config, device=device)

    sequences = [seqs.to(device) for seqs in torch.load(args.data)]
    sequences_x_train, \
    sequences_y_train, \
    sequences_x_test, \
    sequences_y_test = sequences

    if args.loss_fn == "cross_entropy":
        assert args.output_dim == sequences_y_train.max() + 1, "Number of classes in data doesn't match output dimension"
        assert args.output_dim > 1, "Output dimension must be greater than 1 for cross-entropy loss"

    set_seeds(args.seed) # set seed for training (batching, etc.)
    train_loader = FastTensorDataLoader(sequences_x_train, sequences_y_train, batch_size=args.train_batch_size, shuffle=True)
    train_loader = cycle(train_loader)
    test_loader = FastTensorDataLoader(sequences_x_test, sequences_y_test, batch_size=args.test_batch_size, shuffle=True)
    test_loader = cycle(test_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss_fn == "cross_entropy":
        loss_fn = lambda y_pred, y_target: F.cross_entropy(y_pred, y_target, reduction="none")
    elif args.loss_fn == "mse":
        loss_fn = lambda y_pred, y_target: 0.5 * torch.pow(y_pred - y_target, 2)
    elif args.loss_fn == "log":
        loss_fn = lambda y_pred, y_target: 0.5 * torch.log(1 + torch.pow(y_pred - y_target, 2))

    os.makedirs(args.save_dir, exist_ok=True)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_steps = []
    test_steps = []
    pbar = tqdm(total=args.steps, disable=not args.progress_bar)
    for step in range(args.steps):
        if step % 100 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                x, y_target = next(test_loader)
                x, y_target = prepare_xy(x, y_target, args.vectorize_input, args.input_dim, dtype, args.loss_fn)
                y_pred, _ = model.forward_sequence(x)
                if args.loss_fn == "cross_entropy":
                    y_pred = y_pred.transpose(1, 2) # cross entropy expects (batch_size, num_classes, seq_len)
                batch_losses = loss_fn(y_pred, y_target)
                loss = torch.mean(batch_losses[:, args.ignore_first_n_elements:])
                test_losses.append(loss.item())
                if args.loss_fn == "cross_entropy":
                    accuracy = (torch.argmax(y_pred, dim=1) == y_target).float().mean()
                else:
                    accuracy = (torch.round(y_pred) == y_target).float().mean()
                test_accuracies.append(accuracy.item())
                test_steps.append(step)
                if accuracy.item() == 1.0:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_perfect.pt"))
        model.train()
        model.zero_grad()
        x, y_target = next(train_loader)
        x, y_target = prepare_xy(x, y_target, args.vectorize_input, args.input_dim, dtype, args.loss_fn)
        y_pred, _ = model.forward_sequence(x)
        if args.loss_fn == "cross_entropy":
            y_pred = y_pred.transpose(1, 2) # cross entropy expects (batch_size, num_classes, seq_len)
        batch_losses = loss_fn(y_pred, y_target)
        loss = torch.mean(batch_losses[:, args.ignore_first_n_elements:])
        train_losses.append(loss.item())
        if args.loss_fn == "cross_entropy":
            accuracy = (torch.argmax(y_pred, dim=1) == y_target).float().mean()
        else:
            accuracy = (torch.round(y_pred) == y_target).float().mean()
        train_accuracies.append(accuracy.item())
        train_steps.append(step)
        loss.backward()
        optimizer.step()
        tr_l = train_losses[-1]
        te_l = test_losses[-1] if len(test_losses) > 0 else np.nan
        pbar.set_description(f"tr l {tr_l:.2e} te l {te_l:.2e}")
        pbar.update(1)
    pbar.close()

    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_steps": train_steps,
        "test_steps": test_steps
    }

    torch.save(metrics, os.path.join(args.save_dir, "metrics.pt"))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))
    torch.save(model_config, os.path.join(args.save_dir, "model_config.pt"))
    torch.save(vars(args), os.path.join(args.save_dir, "args.pt"))
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f) # also save as json for easier reading
    with open(os.path.join(args.save_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a GeneralRNN")
    parser.add_argument('--data', type=str, help='Path to dataset, a .pt file')
    parser.add_argument('--loss_fn', type=str, default="mse", help='Either "mse" or "log" or "cross_entropy"')
    parser.add_argument('--vectorize_input', action="store_true", help="Convert input ints to one-hot vectors")
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden layer dimension')
    parser.add_argument('--hidden_mlp_depth', type=int, default=1, help='Depth of hidden MLP')
    parser.add_argument('--hidden_mlp_width', type=int, default=16, help='Width of hidden MLP')
    parser.add_argument('--output_mlp_depth', type=int, default=1, help='Depth of output MLP')
    parser.add_argument('--output_mlp_width', type=int, default=16, help='Width of output MLP')
    parser.add_argument('--activation', type=str, default="ReLU", help='Activation function')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8192, help='Batch size for testing')
    parser.add_argument('--ignore_first_n_elements', type=int, default=0, help='Ignore loss at first n sequence positions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dtype', type=str, default="float32", help='Pytorch default dtype')
    # parser.add_argument("--slurm", action="store_true", help="Enable parallel training with slurm")
    parser.add_argument('--save_dir', type=str, default="0", help='Directory to save results')
    parser.add_argument('--progress_bar', action="store_true", help='Show progress bar during training')
    args = parser.parse_args()
    train(args)
