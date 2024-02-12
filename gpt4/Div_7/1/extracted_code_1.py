def compute_y_from_x(x_list):
    # Initialize y as an empty list
    y = []
    # Counter for ones in x
    ones_count = 0
    # Iterate over the x list to compute y
    for i, x_val in enumerate(x_list):
        if x_val == [1]:
            ones_count += 1
            
        # If two ones have occurred, reset the counter and append a one to y
        if ones_count == 2:
            y.append([1])
            ones_count = 0
        else:
            # Otherwise, just append a 0 to y
            y.append([0])
            
    return y

# Define the input and expected output
input_output_pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,0,0,0,0,1,0,0,1,1]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,1,1]),
    ([1,0,1,1,1,1,1,1,1,1], [0,0,0,0,1,1,0,1,1,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,0,0,1,0,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,0,0,1,0,0,0,1])
]

# Process each pair and print the result
for ix, (x, expected_y) in enumerate(input_output_pairs):
    computed_y = compute_y_from_x([[i] for i in x])
    if computed_y == [[i] for i in expected_y]:
        print(f'Row {ix+1} Success')
    else:
        print(f'Row {ix+1} Failure')