import itertools

def apply_operations(ops, x):
    # Assuming the function f(x) = op1(x) op op2(x), where op1 and op2 are some operations
    result = []
    op1, op2 = ops
    for xi in x:
        try:
            result.append(eval(f"op1({xi}) {op2}"))
        except:
            result.append(None)
    return result

def deduce_and_check(x_lists, y_lists):
    # Define a few simple operations to try
    operations = [lambda x: x, lambda x: x + 1, lambda x: x - 1, lambda x: x * 2, lambda x: abs(x)]
    operation_symbols = ['+', '-', '*', '%']

    # Attempt to apply each pair of operations and check if they match the pattern
    for x, y in zip(x_lists, y_lists):
        for op1, op2 in itertools.product(operations, operations):
            for symbol in operation_symbols:
                ops = (op1, symbol + "op2(xi)")
                calculated_y = apply_operations(ops, x)

                # Check if the calculated list matches the given y
                if calculated_y == y:
                    print('Success:', ops)
                    return ops  # Return success if a matching operation is found

    print('Failure')  # No matching operation was found after searching
    return None

# List of x and y values for the first 5 rows
x_lists = [
    [1, 2, 4, 5, 4, 3, 5, 5, 0, 0],
    [0, 5, 6, 4, 1, 2, 6, 6, 2, 3],
    [1, 0, 0, 3, 3, 5, 1, 1, 2, 2],
    [2, 5, 0, 0, 5, 0, 4, 3, 6, 1],
    [0, 1, 5, 5, 6, 4, 0, 2, 5, 5]
]

y_lists = [
    [1, 3, 0, 5, 2, 5, 3, 1, 1, 1],
    [0, 5, 4, 1, 2, 4, 3, 2, 4, 0],
    [1, 1, 1, 4, 0, 5, 6, 0, 2, 4],
    [2, 0, 0, 0, 5, 5, 2, 5, 4, 5],
    [0, 1, 6, 4, 3, 0, 0, 2, 0, 5]
]

# Deduce the operation and check the first 5 rows
deduced_ops = deduce_and_check(x_lists, y_lists)