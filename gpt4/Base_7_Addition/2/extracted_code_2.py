import itertools

# Here are the first five rows of x and y from the table
list_x = [
    [1, 2], [4, 5], [4, 3], [5, 5], [0, 0], [0, 5], [6, 4], [1, 2], [6, 6], [2, 3],
    [1, 0], [0, 3], [3, 5], [1, 1], [2, 2], [2, 5], [0, 0], [5, 0], [4, 3], [6, 1],
    [0, 1], [5, 5], [6, 4], [0, 2], [5, 5], [1, 2], [5, 5], [1, 4], [6, 5], [3, 0],
    [5, 0], [2, 5], [5, 2], [4, 2], [6, 0], [4, 4], [3, 5], [0, 5], [3, 2], [6, 3],
    [0, 1], [6, 3], [4, 2], [6, 5], [4, 5], [5, 4], [1, 5], [5, 1], [4, 3], [4, 6]
]
list_y = [
    3, 2, 1, 4, 1, 5, 3, 4, 5, 6,
    1, 3, 1, 3, 4, 0, 1, 5, 0, 1,
    1, 3, 4, 3, 3, 4, 3, 6, 4, 4,
    5, 0, 1, 0, 0, 2, 2, 6, 5, 2,
    1, 2, 0, 5, 3, 3, 0, 0, 1, 4
]

# Now we need to find the rule to compute y based on x.
# This is a brute force attempt checking some basic arithmetic operations.

def compute_y(x, operation):
    a, b = x
    if operation == 'sum':
        return a + b
    elif operation == 'diff':
        return abs(a - b)
    elif operation == 'product':
        return a * b
    elif operation == 'max':
        return max(a, b)
    elif operation == 'min':
        return min(a, b)
    
# These are the operations we are testing.
operations = ['sum', 'diff', 'product', 'max', 'min']

# Now we loop through each operation and check if we get the expected y list.
success = False
for op in operations:
    generated_y = [compute_y(pair, op) for pair in itertools.chain(*[list_x[i * 10:(i + 1) * 10] for i in range(5)])]
    if generated_y == list_y:
        print(f'Success with operation: {op}')
        success = True
        break

if not success:
    print('Failure: No operation matched the expected list y.')