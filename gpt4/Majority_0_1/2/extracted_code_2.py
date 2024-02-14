# Define the function to compute 'list y' from 'list x'
def compute_y(list_x):
    list_y = [0] * len(list_x)  # Initialize 'list y' with all zeros
    
    for i in range(len(list_x)):
        # Check if the current element is 1 and one of the conditions is met
        if list_x[i] == 1 and (i == 0 or list_x[i - 1] == 1 or (i >= 3 and (list_x[i - 1] == 1 or list_x[i - 2] == 1 or list_x[i - 3] == 1))):
            list_y[i] = 1
    return list_y

# Define the input data (first 5 rows)
input_data = [
    ([0,1,0,0,0,1,0,0,0,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [1,0,1,1,1,1,1,1,1,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,0,1,0,1,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,0,1,1,1,1,1,1]),
]

# Loop through each data row and check the computed 'list y' against the expected one
for list_x, expected_y in input_data:
    computed_y = compute_y(list_x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')