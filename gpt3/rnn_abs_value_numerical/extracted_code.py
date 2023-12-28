# Define the lists
x_lists = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

# Expected output lists y
y_lists_expected = [
    [42, 33, 24, 86, 74, 35, 80, 24, 50, 13],
    [78, 86, 90, 54, 69, 72, 15, 5, 67, 6],
    [49, 24, 27, 11, 1, 87, 41, 69, 13, 81],
    [72, 20, 25, 29, 33, 64, 39, 76, 68, 90],
    [86, 78, 77, 19, 7, 23, 43, 6, 7, 23]
]

# Compute list y and compare with the expected output
for i in range(5):
    y = [abs(num) for num in x_lists[i]]
    if y == y_lists_expected[i]:
        print(f'Row {i+1}: Success')
    else:
        print(f'Row {i+1}: Failure')