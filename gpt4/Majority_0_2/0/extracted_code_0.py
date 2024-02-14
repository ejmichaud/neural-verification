# A possible hypothesis could be the presence of a pattern tied to the index
# Let's attempt using modulo based on index as a simple guess

def compute_y(x):
    # Initial guess at a pattern: output depends on the index position's modulo value
    return [x[i] % 3 if i % 3 == 2 else x[i] % 2 for i in range(len(x))]

# Data from the first 5 rows provided
list_xs = [
    [0, 2, 1, 1, 0, 2, 1, 2, 1, 2],
    [1, 1, 2, 0, 0, 1, 2, 1, 0, 1],
    [1, 2, 1, 2, 2, 1, 2, 0, 1, 1],
    [0, 0, 1, 1, 2, 1, 2, 2, 1, 0],
    [2, 0, 1, 0, 0, 2, 2, 1, 0, 2]
]

list_ys = [
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# Apply the guessed function and compare with the provided y lists
for i, x in enumerate(list_xs):
    computed_y = compute_y(x)
    if computed_y == list_ys[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')
        print(f'Computed: {computed_y}')
        print(f'Expected: {list_ys[i]}')