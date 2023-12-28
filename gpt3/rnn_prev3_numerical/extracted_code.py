def compute_y_from_x(x):
    y = [0, 0, 0]
    y.extend(x[:-3])
    return y

# Input data
x_values = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Expected output
expected_y_values = [
    [0,0,0,42,67,76,14,26,35,20],
    [0,0,0,78,14,10,54,31,72,15],
    [0,0,0,49,76,73,11,99,13,41],
    [0,0,0,72,80,75,29,33,64,39],
    [0,0,0,86,22,77,19,7,23,43]
]

# Compute y from x and check if the output matches the expected values
for i in range(5):
    computed_y = compute_y_from_x(x_values[i])
    if computed_y == expected_y_values[i]:
        print("Success")
    else:
        print("Failure")