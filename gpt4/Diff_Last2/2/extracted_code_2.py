def transform_x_to_y(x_list):
    y_list = []
    y_list.append(x_list[0])  # y_1 = x_1
    y_list.append(x_list[1] - x_list[0])  # y_2 = x_2 - x_1
    y_list.append(x_list[2] + 33)  # y_3 = x_3 + 33
    y_list.append(x_list[3] - (2 * x_list[0]))  # y_4 = x_4 - 2*x_1
    y_list.append(x_list[4] - 62)  # y_5 = x_5 - 62
    return y_list

# Define the first 5 rows of x and y
x_values = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

y_values = [
    [42, -75, 9, -62, 12, 109, -115, 104, 26, -37],
    [78, -164, -4, 144, -123, 141, -57, -20, 72, -61],
    [49, -73, -3, 38, -12, -86, 128, 28, -82, -68],
    [72, -92, -5, 54, 4, 31, -25, 37, -144, -22],
    [86, -164, 155, -58, -12, 16, 20, -49, -1, -16]
]

# Process each row and check for success or failure
for i in range(len(x_values)):
    # Transform x to y
    y_computed = transform_x_to_y(x_values[i])
    # Check first five elements since we're only transforming the first five
    if y_computed == y_values[i][:5]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")