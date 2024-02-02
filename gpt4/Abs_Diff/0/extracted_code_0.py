def compute_y(x):
    y = []
    for i in range(len(x)):
        next_index = (i + 1) % len(x)  # This will wrap around to the first element for the last element in x
        y_value = x[i] + abs(x[next_index])
        y.append(y_value)
    return y

# Define the first five rows of x lists (as tuples for immutability)
x_lists = (
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23],
)

# Define the corresponding y lists
y_lists = (
    [42, 75, 9, 62, 12, 109, 115, 104, 26, 37],
    [78, 164, 4, 144, 123, 141, 57, 20, 72, 61],
    [49, 73, 3, 38, 12, 86, 128, 28, 82, 68],
    [72, 92, 5, 54, 4, 31, 25, 37, 144, 22],
    [86, 164, 155, 58, 12, 16, 20, 49, 1, 16],
)

# Check computed y against the actual y lists and print 'Success' or 'Failure'
for i in range(len(x_lists)):
    computed_y = compute_y(x_lists[i])
    if computed_y == list(y_lists[i]):
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')