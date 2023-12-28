def compute_y(x):
    y = []
    # A simple assumed function, where y[i] = NOT(x[i]) if x[i-1] was 1, else y[i] = x[i]
    for i in range(len(x)):
        if i > 0 and x[i-1] == 1:
            y.append(int(not x[i]))
        else:
            y.append(x[i])
    return y

# Given list pairs
list_pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,1,1,1,1,1,1,1,1]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,1,1,0,1,1,1]),
    ([1,0,1,1,1,1,1,1,1,1], [1,1,0,1,1,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,1,0,1,1,1,0,1,1]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,1,0,1,0,0,1,1,1]),
]

# Apply the compute_y function and check against provided y
for x, expected_y in list_pairs:
    calculated_y = compute_y(x)
    if calculated_y == expected_y:
        print(f'Success for x = {x}, y = {calculated_y}')
    else:
        print(f'Failure: expected {expected_y}, but calculated {calculated_y}')