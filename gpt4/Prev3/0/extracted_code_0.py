# Define the input lists for x
input_lists_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Define the expected output lists for y
expected_output_lists_y = [
    [0,0,0,42,67,76,14,26,35,20],
    [0,0,0,78,14,10,54,31,72,15],
    [0,0,0,49,76,73,11,99,13,41],
    [0,0,0,72,80,75,29,33,64,39],
    [0,0,0,86,22,77,19,7,23,43]
]

# Define a function to compute the y list from the x list
def compute_y_from_x(x):
    return [0, 0, 0] + x[:7]

# Check the output for each list
for i in range(5):  # Iterate over the first five input lists
    computed_y = compute_y_from_x(input_lists_x[i])  # Compute list y
    if computed_y == expected_output_lists_y[i]:  # Check if the computed list matches the expected list
        print(f'Row {i+1}: Success')
    else:  # If they do not match, print Failure
        print(f'Row {i+1}: Failure')