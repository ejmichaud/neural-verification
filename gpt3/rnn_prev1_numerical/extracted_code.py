def compute_y_from_x(x):
    y = [0]  # Initialize list y with 0 as the first element
    for i in range(1, len(x)+1):
        y.append(y[i-1] + x[i-1])  # Compute each element in y based on the previous element and the corresponding element in x
    return y

# Define the given lists
x_lists = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

y_lists = [
    [0,42,67,76,14,26,35,20,24,50],
    [0,78,14,10,54,31,72,15,95,67],
    [0,49,76,73,11,99,13,41,69,87,19],
    [0,72,80,75,29,33,64,39,76,32],
    [0,86,22,77,19,7,23,43,94,93,77]
]

# Compute list y from list x and check the output
for i in range(5):
    computed_y = compute_y_from_x(x_lists[i])
    if computed_y == y_lists[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")