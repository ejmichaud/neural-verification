# Given lists of x
lists_x = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

# Given lists of y
lists_y = [
    [42, 33, 24, 86, 74, 35, 80, 24, 50, 13],
    [78, 86, 90, 54, 69, 72, 15, 5, 67, 6],
    [49, 24, 27, 11, 1, 87, 41, 69, 13, 81],
    [72, 20, 25, 29, 33, 64, 39, 76, 68, 90],
    [86, 78, 77, 19, 7, 23, 43, 6, 7, 23]
]

# Function to compute list y from list x
def compute_y_from_x(list_x):
    return [abs(x) for x in list_x]

# Check if the computed list y matches the given list y
for i in range(5):
    computed_y = compute_y_from_x(lists_x[i])
    if computed_y == lists_y[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")