def compute_y(x):
    y = [0] * len(x)  # Initialize y with zeros
    for i in range(1, len(x) - 1):
        y[i] = 1 if x[i] == 1 and x[i - 1] == 0 and x[i + 1] == 0 else 0
    return y

# Given data
data = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,0,0,0,0,0,0])
]

# Check the output
for idx, (x, expected_y) in enumerate(data):
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print(f"Row {idx + 1}: Success")
    else:
        print(f"Row {idx + 1}: Failure")