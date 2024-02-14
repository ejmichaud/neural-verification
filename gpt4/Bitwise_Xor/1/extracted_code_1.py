# Define a function that applies the hypothesis rule
def compute_y(x_pair):
    # Check if exactly one element in the pair is 1 (our hypothesis)
    return int(sum(x_pair) == 1)

# Given data from the table for the first five rows
x_lists = [
    [[0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [0, 1]],
    [[0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]]
]
y_lists = [
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
]

# Check if the generated y matches the given y for first five rows
for i, x_list in enumerate(x_lists):
    computed_y_list = [compute_y(x_pair) for x_pair in x_list]
    if computed_y_list == y_lists[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')