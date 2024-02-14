def compute_y(x):
    return [0] + x

# Define the first 5 rows of x and y from the table
x_rows = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

y_rows = [
    [0, 42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [0, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [0, 49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [0, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [0, 86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Iterate through the first 5 rows and compute y from x, then check if they match
for i in range(5):
    computed_y = compute_y(x_rows[i])
    if computed_y == y_rows[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")