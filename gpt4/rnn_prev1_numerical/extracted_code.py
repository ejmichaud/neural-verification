# Define the function to shift the list and prepend a 0
def compute_y_from_x(x):
    return [0] + x[:-1]

# Define the input and expected output for the first 5 rows
data = [
    ([42,67,76,14,26,35,20,24,50,13], [0,42,67,76,14,26,35,20,24,50]),
    ([78,14,10,54,31,72,15,95,67,6], [0,78,14,10,54,31,72,15,95,67]),
    ([49,76,73,11,99,13,41,69,87,19], [0,49,76,73,11,99,13,41,69,87]),
    ([72,80,75,29,33,64,39,76,32,10], [0,72,80,75,29,33,64,39,76,32]),
    ([86,22,77,19,7,23,43,94,93,77], [0,86,22,77,19,7,23,43,94,93]),
]

# Loop through each row, compute list y from list x, and verify the result
for x_list, expected_y_list in data:
    computed_y_list = compute_y_from_x(x_list)
    if computed_y_list == expected_y_list:
        print('Success')
    else:
        print('Failure')