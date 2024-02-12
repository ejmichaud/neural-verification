def compute_y(x):
    # Initialize y list
    y = []
    
    # Compute y based on observed pattern
    for i in range(len(x)):
        # Subtract twice the (index+1) from each x element
        y.append(x[i] - (i + 1) * 2)
    return y

# Define the first 5 rows of list x
x_lists = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23],
]

# Define the first 5 rows of list y
y_lists = [
    [42, -9, -9, 62, -12, -39, 45, -56, 26, -37],
    [78, 8, 4, -36, 15, 3, -57, -10, 62, -61],
    [49, -25, 3, -16, -10, 86, -46, 28, -56, 68],
    [72, -52, 5, 4, 4, 31, -25, 37, -8, 22],
    [86, -8, -1, -58, -12, 16, 20, -37, 1, 16],
]

# Iterate over the first 5 rows and compute y from x. Then compare to the given y list.
for i in range(5):
    computed_y = compute_y(x_lists[i])
    if computed_y == y_lists[i]:
        print("Row {} comparison: Success".format(i+1))
    else:
        print("Row {} comparison: Failure".format(i+1))