def calculate_y(x):
    y = [x[0] - 1]  # Start with x[0] - 1 for the first element
    for i in range(1, len(x)):
        y.append(y[-1] + x[i] + 1)  # y[n] = y[n-1] + x[n] + 1
    return y

# Define the first five rows of list x and list y
x_values = [
    [5, -8, -6, 9, 8, 7, 9, -5, -3, 4],
    [-3, 9, 10, 8, 5, 6, 10, 3, -1, 0],
    [-9, 4, -3, 7, 7, 9, -2, 5, 6, 6],
    [-1, 2, -3, -3, -5, -3, 1, 7, 3, 5],
    [4, 5, 9, 2, -4, 1, 4, 6, 2, -5],
]

y_values = [
    [4, -1, -13, -17, -14, -5, 12, 23, 30, 40],
    [-4, 0, 13, 33, 57, 86, 124, 164, 202, 239],
    [-10, -17, -28, -33, -32, -23, -17, -7, 8, 28],
    [-2, -3, -8, -17, -32, -51, -70, -83, -94, -101],
    [3, 10, 25, 41, 52, 63, 77, 96, 116, 130],
]

# Compute y from x for each row and check if it matches the given y
for i in range(len(x_values)):
    computed_y = calculate_y(x_values[i])
    if computed_y == y_values[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")