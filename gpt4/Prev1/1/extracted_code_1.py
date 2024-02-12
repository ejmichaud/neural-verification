# Define the lists of x
lists_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Define the actual lists of y for comparison
lists_y = [
    [0, 42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [0, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [0, 49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [0, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [0, 86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Define the function to compute list y from list x
def compute_y(x):
    return [0] + x

# Check if the function's output matches the actual list y
for i, x in enumerate(lists_x):
    computed_y = compute_y(x)
    if computed_y == lists_y[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")