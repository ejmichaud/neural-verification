
"""
This script attempts to find the simplest RNN that can solve a given task.

Scheme for architecture search:

We are exploring a 5d experiment space:
(hidden_dim, hidden_mlp_depth, hidden_mlp_width, output_mlp_depth, output_mlp_width)
We care most about hidden_dim. Our goal is to find the network with the smallest
hidden dim that can get 100% test accuracy on the task.

Increasing each parameter, holding others constant, strictly increases the expressivity
of the network. However, we don't have a total ordering, just a partial ordering
of expressivity. For instance, on some tasks a network with hidden_dim=8 and
hidden_mlp_depth=1 might be more expressive than a network with hidden_dim=4 and
hidden_mlp_depth=2. So we can't just do a binary search over some total ordering.

We do have a partial ordering. We can describe it as a DAG. We can do a topological

the plan: give a total ordering by (1) specifying finite ranges for each of the 
parameters and (2) giving an ordering of the parameters (e.g. hidden_dim strictly 
most important, then _ then _ then _ then _). Then make functions for converting 
between integers and tuples. Then do binary search on the space of integers, 
starting somewhere in the early-middle. Choose the starting point in a smart way. 
For instance, if we start at 0, then doing an exponential increase will spend a bunch
 of time exploring hidden_dim=1. Start at a point where hidden_dim is higher than 1. 

I've just had a realization that actually smaller networks might be able to generalize
better than larger networks. So expressivity isn't what we care about (because
the generalization gap might be nonzero). So I think I should just do something
more manual. Maybe start with a where we do a binary search over hidden_dims

Okay so actually I think we can keep our scheme from before but just explore the space
from the bottom up. So we start with 0, 1, 2, 4, 8, 16, etc. Once we hit point where
we get 100% test accuracy, we do a binary search bertween that and the previous
point which had failed. We keep doing this until we find the smallest network
that can solve the task.
    this isn't good right now since our ordering of the parameters will mean that
    we will explore all the range of depths and widths of the MLPs at hidden_dim=1
    before we explore hidden_dim.
        we could try to get around this by changing the order such that hidden_dim
        was the least important. but then if depth > 1 are required, we will explore
        the full range of hidden_dims and then when we move beyond this we will explore
        higher depth but always at hidden_dim=max. but we might want to discover
        networks of depth > 1 but with hidden_dim << max!

        we could instead order by parameter count, breaking ties like we are now
        (lexicographically?)

            Let's work out the formula for parameter count versus 
            (h, dh, wh, do, wo), also includingo, the output dimension.
            
            The number of parameters in the hidden mlp is 
            (h * wh + wh) + (wh * wh + wh) * (dh - 2) + (wh * h + h)
            The number of parameters in the output mlp is
            (h * wo + wo) + (wo * wo + wo) * (do - 2) + (wo * o + o)

            
            ,  is:

                (h * wh + 1) * dh + (h * do + 1) * wo   

<scratch this>
Okay so first do some exploration of linear RNNs. Vary hidden_dim. If we find
a width that generalizes, decrease the width and find the smallest width that
generalizes. If no widths generalize, then increase the depth of both MLPs. 
Play with their width a bit. If we generalize perfectly, then decrease the width
as much as possible for both. Then try decreasing the depth of each
</scratch this>

TODO: organize into a main function which takes just args as a parameter

"""

import os
import sys
import time
import random
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from rnn_train import train

def range_product(ranges):
    """
    Compute the product of the ranges.
    ranges: A list of ranges for each parameter.
    """
    result = 1
    for r in ranges:
        result *= (r[1] - r[0] + 1)
    return result

def tuple_to_int(t, ranges):
    """
    Map a tuple to an integer.
    t: The tuple representing the parameters.
    ranges: A list of ranges for each parameter.
    """
    offset = 0
    multiplier = 1
    for value, r in zip(reversed(t), reversed(ranges)):
        offset += (value - r[0]) * multiplier
        multiplier *= (r[1] - r[0] + 1)
    return offset

def int_to_tuple(i, ranges):
    """
    Map an integer to a tuple.
    i: The integer to be mapped.
    ranges: A list of ranges for each parameter.
    """
    rp = range_product(ranges)
    if i >= rp:
        i = rp - 1
    result = []
    for r in reversed(ranges):
        range_size = r[1] - r[0] + 1
        result.append(i % range_size + r[0])
        i //= range_size
    return tuple(reversed(result))

def equivalent(n1, n2, ranges):
    """Checks if two int representations of network shapes correspond 
    to equivalent networks: there is degeneracy in the shape -> network
    map, notably that the widths dont' matter if depth = 1"""
    t1 = int_to_tuple(n1, ranges)
    t2 = int_to_tuple(n2, ranges)
    result = True
    result = result and t1[0] == t2[0]
    result = result and t1[1] == t2[1]
    result = result and t1[2] == t2[2]
    if t1[0] == t2[0] == 1:
        result = result
    else:
        result = result and t1[3] == t2[3]
    if t1[1] == t2[1] == 1:
        result = result
    else:
        result = result and t1[4] == t2[4]
    return result

GREEN = "\033[92m"
RED = "\033[91m"
DEFAULT = "\033[0m"
GREY = "\033[90m"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs architecture search over GeneralRNN shape")
    parser.add_argument('--data', type=str, help='Path to dataset, a .pt file')
    parser.add_argument('--loss_fn', type=str, default="mse", help='Either "mse" or "log" or "cross_entropy"')
    parser.add_argument('--vectorize_input', action="store_true", help="Convert input ints to one-hot vectors")
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--max_hidden_dim', type=int, default=128, help='Max hidden layer dimension')
    parser.add_argument('--max_hidden_mlp_depth', type=int, default=3, help='Max depth of hidden MLP')
    parser.add_argument('--max_hidden_mlp_width', type=int, default=256, help='Max width of hidden MLP')
    parser.add_argument('--max_output_mlp_depth', type=int, default=3, help='Max depth of output MLP')
    parser.add_argument('--max_output_mlp_width', type=int, default=256, help='Max width of output MLP')
    parser.add_argument('--activation', type=str, default="ReLU", help='Activation function')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--train_batch_size', type=int, default=4096, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=65536, help='Batch size for testing')
    parser.add_argument('--ignore_first_n_elements', type=int, default=0, help='Ignore loss at first n sequence positions')
    parser.add_argument('--dtype', type=str, default="float32", help='Pytorch default dtype')
    # parser.add_argument('--slurm', action="store_true", help="Enable parallel training with slurm")
    # parser.add_argument('--max_simultaneous_runs', type=int, default=10, help="Max number of slurm jobs to run at once")
    parser.add_argument('--seeds-per-run', type=int, default=3, help="Number of seeds per run")
    parser.add_argument('--save_dir', type=str, default="0", help='Directory to save results')
    # parser.add_argument('--verbose', action="store_true", help='Show progress bar during training')
    args = parser.parse_args()

    range_hidden_dim = (1, args.max_hidden_dim)
    range_hidden_mlp_depth = (1, args.max_hidden_mlp_depth)
    range_hidden_mlp_width = (1, args.max_hidden_mlp_width)
    range_output_mlp_depth = (1, args.max_output_mlp_depth)
    range_output_mlp_width = (1, args.max_output_mlp_width)
    
    # this ordering determines the total ordering. variables
    # occurring earlier are strictly more important than 
    # variables occurring later
    # let's try actually putting the depths first.
    ranges = [
        range_output_mlp_depth,
        range_hidden_mlp_depth,
        range_hidden_dim,
        range_output_mlp_width,
        range_hidden_mlp_width,
    ]
    # Size of the architecture space
    N = range_product(ranges)
    print(f"log2( architecture search space ) = {np.log2(N)}")

    # initiial architecture
    nwo = range_output_mlp_width[1] - range_output_mlp_width[0] + 1
    nwh = range_hidden_mlp_width[1] - range_hidden_mlp_width[0] + 1
    # with our ordering of the parameters, the first nwo * nwh
    # elements are all the same since when depth = 1 the width
    # of the MLPs doesn't matter. so we skip these elements. 
    # if nwo * nwh is a power of 2, then as we increase n by
    # factors of 2, we will first explore all the hidden_dim sizes
    # at depth = 1, and then next we will explore higher depths
    n = nwo * nwh

    RUN_RECORDS = []
    
    prev_n = -1
    success_n = N
    success_seed = None
    rounds = 0

    # do initial run until n = N or we find a network that works
    while n <= N * 2 and not success_seed:
        if equivalent(n, prev_n, ranges):
            prev_n = n
            n *= 2
            continue
        seeds = list(range(args.seeds_per_run))
        args_run = argparse.Namespace(**vars(args).copy())
        args_run.output_mlp_depth, \
        args_run.hidden_mlp_depth, \
        args_run.hidden_dim, \
        args_run.output_mlp_width, \
        args_run.hidden_mlp_width = int_to_tuple(n, ranges)
        args_run.progress_bar = False
        sys.stdout.write("(od{:d}, hd{:d}, hdm{:d}, ow{:d}, hw{:d})  ".format(
            args_run.output_mlp_depth,
            args_run.hidden_mlp_depth,
            args_run.hidden_dim,
            args_run.output_mlp_width,
            args_run.hidden_mlp_width,
        ))
        sys.stdout.flush()
        for seed in seeds:
            args_run.seed = seed
            run_dir = os.path.join(args.save_dir, "runs", f"{n}_{seed}")
            args_run.save_dir = run_dir
            train(args_run)
            metrics = torch.load(os.path.join(run_dir, "metrics.pt"))
            if max(metrics['test_accuracies']) == 1.0:
                success_n = n
                success_seed = seed
                RUN_RECORDS.append((vars(args_run), "succeeded"))
                sys.stdout.write(GREEN+"."+DEFAULT)
                sys.stdout.write(GREEN+"  succeeded\n"+DEFAULT)
                sys.stdout.flush()
                break
            else:
                RUN_RECORDS.append((vars(args_run), "failed"))
                sys.stdout.write(RED+"."+DEFAULT)
                sys.stdout.flush()
                
        if success_seed is None:
            prev_n = n
            n *= 2
            sys.stdout.write(RED+"  failed\n"+DEFAULT)
        else:
            break
        rounds += 1

    if success_seed is None:
        print("Failed to find a network that works")
        df = pd.DataFrame(RUN_RECORDS, columns=["args", "outcome"])
        df.to_csv(os.path.join(args.save_dir, "run_records.csv"))
        exit()
    
    # now do a binary search between success_n and n // 2
    prev_n = success_n
    failed_n = n // 2
    success_run_args = None
    while success_n - failed_n > 1:
        n = (success_n + failed_n) // 2
        seeds = list(range(args.seeds_per_run))
        args_run = argparse.Namespace(**vars(args).copy())
        args_run.output_mlp_depth, \
        args_run.hidden_mlp_depth, \
        args_run.hidden_dim, \
        args_run.output_mlp_width, \
        args_run.hidden_mlp_width = int_to_tuple(n, ranges)
        args_run.progress_bar = False
        sys.stdout.write("(od{:d}, hd{:d}, hdm{:d}, ow{:d}, hw{:d})  ".format(
            args_run.output_mlp_depth,
            args_run.hidden_mlp_depth,
            args_run.hidden_dim,
            args_run.output_mlp_width,
            args_run.hidden_mlp_width,
        ))
        sys.stdout.flush()
        if equivalent(n, prev_n, ranges):
            sys.stdout.write(GREY+"skipped (equivalent to previous architecture)\n"+DEFAULT)
            sys.stdout.flush()
            # perform the update where the the outcome is the same
            # as the previous outcome. 
            if prev_n == success_n: # previous outcome was success
                success_n = n # this will be success too
            else: # previous outcome a failure
                failed_n = n # this will be failure too
            prev_n = n
            continue
        for seed in seeds:
            args_run.seed = seed
            run_dir = os.path.join(args.save_dir, "runs", f"{n}_{seed}")
            args_run.save_dir = run_dir
            train(args_run)
            metrics = torch.load(os.path.join(run_dir, "metrics.pt"))
            if max(metrics['test_accuracies']) == 1.0:
                success_n = n
                success_seed = seed
                success_run_args = args_run
                RUN_RECORDS.append((vars(args_run), "succeeded"))
                sys.stdout.write(GREEN+"."+DEFAULT)
                sys.stdout.write(GREEN+"  succeeded\n"+DEFAULT)
                sys.stdout.flush()
                break
            else:
                RUN_RECORDS.append((vars(args_run), "failed"))
                sys.stdout.write(RED+"."+DEFAULT)
                sys.stdout.flush()
        # print_round_report(args_run, n, seed, success_n == n)
        if success_n != n: # if we failed
            failed_n = n
            sys.stdout.write(RED+"  failed\n"+DEFAULT)
        prev_n = n

    print(f"Found smallest network that works: {success_n}")
    print(f"Seed: {success_seed}")
    print(f"Parameters: {int_to_tuple(success_n, ranges)}")
    # save the final run args
    torch.save(success_run_args, os.path.join(args.save_dir, "smallest_run_args.pt"))
    # also save the run records
    df = pd.DataFrame(RUN_RECORDS, columns=["args", "outcome"])
    df.to_csv(os.path.join(args.save_dir, "run_records.csv"))
    exit()

