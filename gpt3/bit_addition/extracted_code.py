# Define the input list x
list_x = [
    [0, 1],[0, 0],[0, 1],[0, 0],[0, 1]
]

# Define the expected list y
list_y_expected = [1, 0, 1, 0, 1]

# Compute list y using the formula
list_y = [(x[0] + x[1]) % 2 for x in list_x]

# Compare the computed list y with the expected list y and print the result
if list_y == list_y_expected:
    print('Success')
else:
    print('Failure')