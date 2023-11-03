
import os
import random
import argparse

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script for GeneralRNN model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden layer dimension')
    parser.add_argument('--hidden_mlp_depth', type=int, default=1, help='Depth of hidden MLP')
    parser.add_argument('--hidden_mlp_width', type=int, default=100, help='Width of hidden MLP')
    parser.add_argument('--output_mlp_depth', type=int, default=1, help='Depth of output MLP')
    parser.add_argument('--output_mlp_width', type=int, default=100, help='Width of output MLP')
    parser.add_argument('--activation', type=str, default="ReLU", help='Activation function')
    parser.add_argument('--steps', type=int, default=1500, help='Number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8192, help='Batch size for testing')
    parser.add_argument('--progress_bar', type=bool, default=True, help='Show progress bar')
    parser.add_argument('--dtype', type=str, default="float32", help='Pytorch default dtype')
    parser.add_argument('--save_dir', type=str, default="0", help='Directory to save results')
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data.pt")
    sequences = [seqs.to(device).to(torch.int64) for seqs in torch.load(data_path)]
    sequences_x_train, \
    sequences_y_train, \
    sequences_x_test, \
    sequences_y_test = sequences

    train_loader = FastTensorDataLoader(sequences_x_train, sequences_y_train, batch_size=args.train_batch_size, shuffle=True)
    train_loader = cycle(train_loader)
    test_loader = FastTensorDataLoader(sequences_x_test, sequences_y_test, batch_size=args.test_batch_size, shuffle=True)
    test_loader = cycle(test_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                x, y_target = x.unsqueeze(2), y_target.unsqueeze(2)
                y_pred, _ = model.forward_sequence(x.to(dtype))
                loss = F.mse_loss(y_pred, y_target.to(dtype))
                test_losses.append(loss.item())
                accuracy = (torch.round(y_pred) == y_target).float().mean()
                test_accuracies.append(accuracy.item())
                test_steps.append(step)
                if accuracy.item() == 1.0:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_perfect.pt")) 
        model.train()
        model.zero_grad()
        x, y_target = next(train_loader)
        x, y_target = x.unsqueeze(2), y_target.unsqueeze(2)
        y_pred, _ = model.forward_sequence(x.to(dtype))
        loss = F.mse_loss(y_pred, y_target.to(dtype))
        train_losses.append(loss.item())
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
