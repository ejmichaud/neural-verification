# A function that computes y from x using the specified transformation
def transform_x_to_y(x):
    y = [0] * len(x)  # Initialize list y with zeros
    for i in range(len(x)):
        if i >= 4:
            y[i] = x[i - 4]
    return y

# List of x values for the first five rows
list_of_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# List of expected y values for the first five rows
expected_list_of_y = [
    [0, 0, 0, 0, 42, 67, 76, 14, 26, 35],
    [0, 0, 0, 0, 78, 14, 10, 54, 31, 72],
    [0, 0, 0, 0, 49, 76, 73, 11, 99, 13],
    [0, 0, 0, 0, 72, 80, 75, 29, 33, 64],
    [0, 0, 0, 0, 86, 22, 77, 19, 7, 23]
]

# Compute y from x and check against the expected y for the first five rows
for x, expected_y in zip(list_of_x, expected_list_of_y): 
    computed_y = transform_x_to_y(x)
    if computed_y == expected_y:
        print('Success')  # The computed y matches the expected y
    else:
        print('Failure')  # There was a mismatch