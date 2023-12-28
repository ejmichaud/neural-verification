def compute_y(x):
    y = [0 if val == 1 else 0 for val in x]
    return y

x_values = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

expected_y_values = [
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0]
]

for i in range(5):
    computed_y = compute_y(x_values[i])
    if computed_y == expected_y_values[i]:
        print("Success")
    else:
        print("Failure")