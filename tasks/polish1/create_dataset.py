"""
----------------------- DATASET DESCRIPTION -----------------------

This dataset is a sequence prediction task where the input is a
sequence of integers and the symbol +, representing an expression
to be evaluated in reverse polish notation. The output at each
sequence position is the element on the top of the stack after
evaluating the expression up to that position. Sequences are of
length 11 (6 integers and 5 +s). Integers are 0-9. 
10 repressents the symbol +.

Author: Eric Michaud

-------------------------------------------------------------------
"""

import os
import random
from tqdm.auto import tqdm
import numpy as np
import torch

def random_rpn_expression(N):
    if N % 2 == 0:
        raise ValueError("N must be odd")

    operand_count = (N + 1) // 2
    operator_count = (N - 1) // 2
    expression = []
    top_of_stack = []
    stack = []

    while operand_count > 0 or operator_count > 0:
        if operand_count > 0 and (len(stack) < 2 or random.choice(['operand', 'operator']) == 'operand'):
            operand = random.randint(0, 9)
            expression.append(operand)
            stack.append(operand)
            top_of_stack.append(operand)
            operand_count -= 1
        elif operator_count > 0:
            expression.append(10) # represents the operator +
            a = stack.pop()  # Pop two operands and push a placeholder result
            b = stack.pop()
            stack.append(a + b)
            top_of_stack.append(a + b)
            operator_count -= 1

    return expression, top_of_stack

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    D = int(1e6)
    split = 0.9

    sequences_x = []
    sequences_y = []

    for i in tqdm(range(D)):
        x, y = random_rpn_expression(11)
        sequences_x.append(x)
        sequences_y.append(y)
    
    # convert to pytorch
    sequences_x = torch.tensor(sequences_x)
    sequences_y = torch.tensor(sequences_y)

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
