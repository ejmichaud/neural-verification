def transform_x_to_y(x_list):
    y_list = x_list.copy()  # Start with a copy of x to preserve the structure

    # Process elements in x_list except for the last element.
    for i in range(len(x_list) - 1):
        # If the current element is 1, make sure the next element is also 1.
        if x_list[i] == 1:
            y_list[i] = 1
            if x_list[i+1] != 1:  # But only if the next element isn't already a 1
                y_list[i+1] = 1
    
    return y_list

# List of example rows for testing
x_rows = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
]

# List of corresponding y rows
y_rows = [
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
]

# Test each row to see if the transformation from x to y is successful
for x_row, expected_y in zip(x_rows, y_rows):
    calculated_y = transform_x_to_y(x_row)
    print('Success' if calculated_y == expected_y else 'Failure')