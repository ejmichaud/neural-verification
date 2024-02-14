def compute_y_from_x(x):
    y = []
    current_max = -float('inf')  # Start with the smallest possible value
    for value in x:
        current_max = max(current_max, value)
        y.append(current_max)
    return y

# Define the first 5 rows of x and y
x_values = [
    [2,7,6,4,6,5,0,4,0,3],
    [8,4,0,4,1,2,5,5,7,6],
    [9,6,3,1,9,3,1,9,7,9],
    [2,0,5,9,3,4,9,6,2,0],
    [6,2,7,9,7,3,3,4,3,7]
]

y_values = [
    [2,7,7,7,7,7,7,7,7,7],
    [8,8,8,8,8,8,8,8,8,8],
    [9,9,9,9,9,9,9,9,9,9],
    [2,2,5,9,9,9,9,9,9,9],
    [6,6,7,9,9,9,9,9,9,9]
]

# Compute y from x and check the output for each row
for i in range(len(x_values)):
    computed_y = compute_y_from_x(x_values[i])
    if computed_y == y_values[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')