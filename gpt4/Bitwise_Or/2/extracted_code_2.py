# Define the x's
x_lists = [
    [[0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 0]],
    [[0, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 0]],
    [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1], [1, 1], [0, 1]],
    [[0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]]
]

# Define the corresponding y's
y_lists = [
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
]

# Placeholder function to compute y from x (to be defined)
def compute_y_from_x(x_list):
    # This is a stub function, as we don't have a clear rule to compute y
    y_computed = []
    for pair in x_list:
        # Assuming a very simple logic where we add the elements of the pair
        # This is just a placeholder and likely incorrect logic
        y_computed.append(sum(pair) % 2)  # using modulus to keep it binary
    return y_computed

# Check each row of x to compute y and compare with the actual y
for i in range(len(x_lists)):
    y_computed = compute_y_from_x(x_lists[i])
    result = "Success" if y_computed == y_lists[i] else "Failure"
    print(f"Row {i+1} check: {result}")
    print(f"Computed y: {y_computed}")
    print(f"Actual y: {y_lists[i]}\n")