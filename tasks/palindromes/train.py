
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from neural_verification import (
    Tokenizer, 
    GPTConfig, 
    GPT, 
    cycle, 
    HeterogeneousDataLoader
)
from config import tokenizer_vocabulary, model_config

def parse_tuple(string):
    """Parses a string representing a tuple of integers"""
    string = string[1:-1]
    split = string.split(",")
    if split[-1] == "":
        split = split[:-1]
    return tuple(map(int, split))

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(model_config).to(device)

    df = pd.read_csv("train.csv")
    strings_train = df["string"].tolist()
    answer_idxs_train = df["answer_idxs"].apply(parse_tuple).tolist()

    df = pd.read_csv("test.csv")
    strings_test = df["string"].tolist()
    answer_idxs_test = df["answer_idxs"].apply(parse_tuple).tolist()

    train_loader = HeterogeneousDataLoader(strings_train, answer_idxs_train, batch_size=1024, shuffle=True)
    train_loader = cycle(train_loader)
    test_loader = HeterogeneousDataLoader(strings_test, answer_idxs_test, batch_size=1024, shuffle=True)
    test_loader = cycle(test_loader)

    steps = 1500
    lr = 3e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    pbar = tqdm(total=steps)
    for step in range(steps):
        model.eval()
        with torch.no_grad():
            batch_strings, batch_answer_idxs = next(test_loader)
            input_ids = tokenizer.encode(batch_strings, padding=True)
            input_ids = torch.tensor(input_ids).reshape((len(batch_strings), -1)).to(device)
            preds = model(input_ids)
            preds_flat = preds[:, :-1, :].reshape(-1, tokenizer.vocab_size)
            input_ids_flat = input_ids[:, 1:].flatten()
            losses = F.cross_entropy(preds_flat, input_ids_flat, reduction="none")
            losses = losses.reshape(len(batch_strings), -1)
            mask = torch.zeros_like(losses)
            for i, answer_idxs_tuple in enumerate(batch_answer_idxs):
                for answer_idx in answer_idxs_tuple:
                    mask[i, answer_idx-1] = 1
            losses = losses * mask
            loss = losses.sum() / mask.sum()
            test_losses.append(loss.item())
        model.train()
        model.zero_grad()
        batch_strings, batch_answer_idxs = next(train_loader)
        input_ids = tokenizer.encode(batch_strings, padding=True)
        input_ids = torch.tensor(input_ids).reshape((len(batch_strings), -1)).to(device)
        preds = model(input_ids)
        preds_flat = preds[:, :-1, :].reshape(-1, tokenizer.vocab_size)
        input_ids_flat = input_ids[:, 1:].flatten()
        losses = F.cross_entropy(preds_flat, input_ids_flat, reduction="none")
        losses = losses.reshape(len(batch_strings), -1)
        mask = torch.zeros_like(losses)
        for i, answer_idxs_tuple in enumerate(batch_answer_idxs):
            for answer_idx in answer_idxs_tuple:
                mask[i, answer_idx-1] = 1
        losses = losses * mask
        loss = losses.sum() / mask.sum()
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pbar.set_description(f"tr l {train_losses[-1]:.2e} te l {test_losses[-1]:.2e}")
        pbar.update(1)
    pbar.close()

    torch.save(model.state_dict(), "model.pt")
