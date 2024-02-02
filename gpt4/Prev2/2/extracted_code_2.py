def compute_y_from_x(x):
    # Initialize list y with zeros
    y = [0] * len(x)
    # Fill in the rest of y, starting from the third element
    y[2:] = x[:-2]
    return y

# Provided test cases
test_cases_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

test_cases_y = [
    [0,0,42,67,76,14,26,35,20,24],
    [0,0,78,14,10,54,31,72,15,95],
    [0,0,49,76,73,11,99,13,41,69],
    [0,0,72,80,75,29,33,64,39,76],
    [0,0,86,22,77,19,7,23,43,94]
]

# Check the computed y against the provided y for each test case
for i, (x, expected_y) in enumerate(zip(test_cases_x, test_cases_y)):
    computed_y = compute_y_from_x(x)
    if computed_y == expected_y:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")