
"""
----------------------- DATASET DESCRIPTION -----------------------

Reverse polish notation.

Author: Anish Mudide

-------------------------------------------------------------------
"""

import os
from math import floor, ceil
import random
import operator
import gzip

import numpy as np
from tqdm.auto import tqdm
import torch

def generate_infix_expression_fixed(length=3):
    """Generate a random infix expression that always ends with a number."""
    ops = ['+', '-', '*', '/']
    expression = [str(random.randint(0, 9))]  # Start with a number

    for _ in range(length - 1):
        op = random.choice(ops)
        num = str(random.randint(0, 9))
        expression.append(op)
        expression.append(num)

    return ' '.join(expression)

def infix_to_rpn(infix_expression):
    """Convert an infix expression to RPN."""
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []

    for token in infix_expression.split():
        if token.isdigit():
            output.append(token)
        elif token in precedence:
            while stack and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        else:
            raise ValueError("Invalid token")

    while stack:
        output.append(stack.pop())

    return ' '.join(output)

def evaluate_expression(expression):
    """Evaluate the arithmetic expression."""
    try:
        return eval(expression)
    except ZeroDivisionError:
        return "Division by zero"

# Define the tokens and their corresponding one-hot encodings
tokens = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
          '+': 10, '-': 11, '*': 12, '/': 13}

def one_hot_encode(rpn_list):
    """Convert a RPN expression list into one-hot encoded vectors."""
    one_hot_encoded = []
    for token in rpn_list:
        vector = [0] * len(tokens)
        if token in tokens:
            vector[tokens[token]] = 1
        one_hot_encoded.append(vector)
    return one_hot_encoded

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    D = int(1e6)
    split = 0.9

    one_hot_encoded_dataset = []
    for _ in range(100):  # Generate 100 expressions
        infix_expr = generate_infix_expression_fixed()
        rpn_expr = infix_to_rpn(infix_expr)
        rpn_list = rpn_expr.split()  # Convert RPN string to a list
        value = evaluate_expression(infix_expr.replace(' ', ''))
        one_hot_rpn = one_hot_encode(rpn_list)
        one_hot_encoded_dataset.append((infix_expr, one_hot_rpn, value))

    # Creating sequences_x and sequences_y datasets
    sequences_x = [one_hot_rpn for _, one_hot_rpn, _ in one_hot_encoded_dataset]
    sequences_y = [value for _, _, value in one_hot_encoded_dataset]

    sequences_x_train = sequences_x[:int(D * split)]
    sequences_x_test = sequences_x[int(D * split):]
    sequences_y_train = sequences_y[:int(D * split)]
    sequences_y_test = sequences_y[int(D * split):]
    # import code; code.interact(local=locals())
    torch.save((
        sequences_x_train, 
        sequences_y_train,
        sequences_x_test, 
        sequences_y_test
    ), os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))
