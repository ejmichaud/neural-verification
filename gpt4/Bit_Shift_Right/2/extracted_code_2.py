def compute_y(x):
    y = [None] * len(x)
    y[0] = x[0]  # Handle the first element specially, as it has no predecessor
    for i in range(1, len(x)):
        y[i] = 0 if x[i-1] == 1 and x[i] == 1 else x[i]
    return y

# Define the first five rows of lists x and y as provided
input_output_pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,0,1,0,0,0,1,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,1,0,1,1,1]),
    ([1,0,1,1,1,1,1,1,1,1], [0,1,0,1,1,1,1,1,1,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,1,1,0,1,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0,1]),
]

# Iterate through the pairs and check if computation is correct
for x_list, expected_y_list in input_output_pairs: 
    # Compute y from x
    computed_y_list = compute_y(x_list)
    
    # Check if the computed y matches the expected y
    if computed_y_list == expected_y_list:
        print('Success')
    else:
        print('Failure')
        print(f'Computed y: {computed_y_list}, Expected y: {expected_y_list}')