def compute_y(x):
    y = []
    for i in range(len(x)):
        # Check if it's a '1' surrounded by '0's or in a sequence of three '1's.
        if x[i] == 1 and ((i == 0 or x[i - 1] == 0) and (i == len(x) - 1 or x[i + 1] == 0)):
            y.append(1)
        else:
            y.append(0)
    return y

# Given lists x and the corresponding y to check the results.
pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,0,0,0,0,0,0])
]

# Check if the computed y matches the given y for the first five rows.
for idx, (x, expected_y) in enumerate(pairs):
    computed_y = compute_y(x)
    if computed_y == expected_y:
        result = 'Success'
    else:
        result = 'Failure'
    print(f'Row {idx+1}: {result}')