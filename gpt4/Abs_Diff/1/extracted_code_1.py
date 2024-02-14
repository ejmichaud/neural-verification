# Define a function to calculate y from x using trial and error for the relation.
def calculate_y(x):
    y = [abs(x[0])]  # The first element of y is the absolute value of the first element of x.

    # Due to the lack of a clear relationship, we will use this block to try and determine
    # a transformation that might fit the rest of the elements. As we don't have a clear pattern,
    # this 'for loop' will simply copy the remaining elements as is without transformation.
    for xi in x[1:]:
        y.append(xi)  # This is a placeholder for a real transformation, if one exists.

    return y

# Given lists x and y for the first 5 rows.
lists_x = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

lists_y = [
    [42, 75, 9, 62, 12, 109, 115, 104, 26, 37],
    [78, 164, 4, 144, 123, 141, 57, 20, 72, 61],
    [49, 73, 3, 38, 12, 86, 128, 28, 82, 68],
    [72, 92, 5, 54, 4, 31, 25, 37, 144, 22],
    [86, 164, 155, 58, 12, 16, 20, 49, 1, 16]
]

# Run the transformation and check if the output matches list y.
for i, x in enumerate(lists_x):
    y_computed = calculate_y(x)
    if y_computed == lists_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')