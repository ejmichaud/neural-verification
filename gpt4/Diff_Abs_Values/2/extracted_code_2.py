def compute_y_from_x(x):
    y = []
    for i in range(len(x)):
        if i == 0:
            # The first element seems to be the same in both lists.
            y.append(x[i])
        else:
            # Computing y elements based on x elements according to the identified pattern
            y_i = x[i] - (2 * x[i-1])
            y.append(y_i)
    return y

# Given the first 5 rows of list x and list y from the provided table
rows_x = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

rows_y = [
    [42, -9, -9, 62, -12, -39, 45, -56, 26, -37],
    [78, 8, 4, -36, 15, 3, -57, -10, 62, -61],
    [49, -25, 3, -16, -10, 86, -46, 28, -56, 68],
    [72, -52, 5, 4, 4, 31, -25, 37, -8, 22],
    [86, -8, -1, -58, -12, 16, 20, -37, 1, 16]
]

# Checking the calculated y list against the given one
for i in range(5):
    calc_y = compute_y_from_x(rows_x[i])
    if calc_y == rows_y[i]:
        print(f"Row {i + 1} - Success")
    else:
        print(f"Row {i + 1} - Failure")