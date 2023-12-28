def compute_y_from_x(x):
    y = []
    for i in range(0, len(x), 2):
        if x[i] == 1 or x[i+1] == 1:
            y.extend([1, 1])
        else:
            y.extend([0, 0])
    return y

# Input data
x = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
]

# Expected output
expected_y = [
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
]

# Compute y and compare with expected output
for i in range(5):
    computed_y = compute_y_from_x(x[i])
    if computed_y == expected_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")