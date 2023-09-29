
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from neural_verification import (
    Tokenizer, 
    TransformerConfig, 
    Transformer, 
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

def compute_loss(input_ids, output, answer_idxs):
    """Computes the mean loss of a transformer's `output` 
    in predicting the next token in `input_ids` at positions `answer_idxs`
    
    :param input_ids: torch.Tensor of shape (batch_size, seq_len)
    :param output: torch.Tensor of shape (batch_size, seq_len, vocab_size)
    :param answer_idxs: list of tuples of integers. Each tuple represents
        the indices of tokens representing the answer in the corresponding
        in the corresponding input_ids sequence.
    """
    output_flat = output[:, :-1, :].reshape(-1, tokenizer.vocab_size)
    input_ids_flat = input_ids[:, 1:].flatten()
    losses = F.cross_entropy(output_flat, input_ids_flat, reduction="none")
    losses = losses.reshape(input_ids.shape[0], -1)
    mask = torch.zeros_like(losses)
    for i, answer_idxs_tuple in enumerate(answer_idxs):
        for answer_idx in answer_idxs_tuple:
            mask[i, answer_idx-1] = 1
    losses = losses * mask
    loss = losses.sum() / mask.sum()
    return loss

@torch.no_grad()
def compute_accuracy(input_ids, output, answer_idxs):
    """Computes the mean accuracy of a transformer's `output`. This is
    the accuracy computed from greedy sampling (zero temperature)
    in predicting the next token in `input_ids` at positions `answer_idxs`.

    Note that this is not in general an exact string match accuracy. It is
    an exact string match when there is only one answer_idx per sequence. 
    However, when the "answer" part of the sequence is spread across multiple 
    tokens, we are computing a per-token accuracy rather than a per-answer
    accuracy. You may want to modify this if your answer is spread across
    multiple tokens.
    
    :param input_ids: torch.Tensor of shape (batch_size, seq_len)
    :param output: torch.Tensor of shape (batch_size, seq_len, vocab_size)
    :param answer_idxs: list of tuples of integers. Each tuple represents
        the indices of tokens representing the answer in the corresponding
        in the corresponding input_ids sequence.
    """
    preds = output.argmax(dim=-1)
    corrects = preds[:, :-1] == input_ids[:, 1:]
    correct_count = 0
    for i, answer_idxs_tuple in enumerate(answer_idxs):
        is_correct = 1
        for answer_idx in answer_idxs_tuple:
            is_correct &= corrects[i, answer_idx-1]
        correct_count += is_correct
    acc = correct_count / input_ids.shape[0]
    return acc

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokenizer = Tokenizer(tokenizer_vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_config).to(device)

    df = pd.read_csv("tasks/kth_root_of_n_multidigit/train.csv")
    inputs_train = df["inputs"].tolist()
    labels_train = df["labels"].astype(str).tolist()
    answer_idxs_train = df["answer_idxs"].apply(parse_tuple).tolist()
    
    print(labels_train[:10])
    df = pd.read_csv("tasks/kth_root_of_n_multidigit/test.csv")
    inputs_test = df["inputs"].tolist()
    labels_test = df["labels"].astype(str).tolist()
    answer_idxs_test = df["answer_idxs"].apply(parse_tuple).tolist()

    train_loader = HeterogeneousDataLoader(inputs_train, labels_train,answer_idxs_train, batch_size=1024, shuffle=True)
    train_loader = cycle(train_loader)
    test_loader = HeterogeneousDataLoader(inputs_test, labels_test,answer_idxs_test, batch_size=1024, shuffle=True)
    test_loader = cycle(test_loader)

    steps = 1500
    lr = 3e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    pbar = tqdm(total=steps)
    for step in range(steps):
        model.eval()
        with torch.no_grad():
            # get batch and forward pass
            batch_inputs, batch_labels, batch_answer_idxs = next(test_loader)
            input_ids = tokenizer.encode(batch_inputs, padding=True)
            input_ids = torch.tensor(input_ids).reshape((len(batch_inputs), -1)).to(device)
            output = model(input_ids)
            label_ids = tokenizer.encode(batch_labels, padding=True)
            label_ids = torch.tensor(label_ids).reshape((len(batch_labels), -1)).to(device)
            # compute metrics
            acc = compute_accuracy(label_ids, output, batch_answer_idxs)
            test_accuracies.append(acc.item())
            loss = compute_loss(label_ids, output, batch_answer_idxs)
            test_losses.append(loss.item())
            
        model.train()
        model.zero_grad()
        # get batch and forward pass
        batch_inputs, batch_labels, batch_answer_idxs = next(train_loader)
        input_ids = tokenizer.encode(batch_inputs, padding=True)
        input_ids = torch.tensor(input_ids).reshape((len(batch_inputs), -1)).to(device)
        output = model(input_ids)
        
        label_ids = tokenizer.encode(batch_labels, padding=True)
        label_ids = torch.tensor(label_ids).reshape((len(batch_labels), -1)).to(device)
        # compute metrics
        acc = compute_accuracy(label_ids, output, batch_answer_idxs)
        train_accuracies.append(acc.item())
        loss = compute_loss(label_ids, output, batch_answer_idxs)
        train_losses.append(loss.item())
        # backprop on loss
        loss.backward()
        optimizer.step()
        pbar.set_description(f"tr l {train_losses[-1]:.2e} te l {test_losses[-1]:.2e}")
        pbar.update(1)
    pbar.close()

    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies
    }

    torch.save(metrics, "tasks/kth_root_of_n_multidigit/metrics.pt")
    torch.save(model.state_dict(), "tasks/kth_root_of_n_multidigit/model.pt")
