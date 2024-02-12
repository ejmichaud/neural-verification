def f(x):
    # Placeholder function, replace with the actual function once known
    return x

# Define the first five rows of list x and list y
list_x = [
    [0, 2, 1, 1, 0, 2, 1, 2, 1, 2],
    [1, 1, 2, 0, 0, 1, 2, 1, 0, 1],
    [1, 2, 1, 2, 2, 1, 2, 0, 1, 1],
    [0, 0, 1, 1, 2, 1, 2, 2, 1, 0],
    [2, 0, 1, 0, 0, 2, 2, 1, 0, 2]
]

list_y = [
    [0, 2, 0, 1, 1, 0, 1, 0, 1, 0],
    [1, 2, 1, 1, 1, 2, 1, 2, 2, 0],
    [1, 0, 1, 0, 2, 0, 2, 2, 0, 1],
    [0, 0, 1, 2, 1, 2, 1, 0, 1, 1],
    [2, 2, 0, 0, 0, 2, 1, 2, 2, 1]
]

# Check if computed list y matches the provided list y
for i, x in enumerate(list_x):
    computed_y = [f(val) for val in x]
    if computed_y == list_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")

    # Print the computed y for verification
    print(f"Computed y[{i+1}]: {computed_y}")
    print(f"Given y[{i+1}]: {list_y[i]}\n")