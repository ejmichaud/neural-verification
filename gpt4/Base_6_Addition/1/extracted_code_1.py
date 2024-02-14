def compute_y(x_list):
    # Hypothesized function (placeholder)
    return (x_list[0] + x_list[1]) % 6

# Initial datasets for the first 5 rows
input_data = [
    [[0, 5], [4, 4], [0, 5], [4, 2], [4, 5], [4, 4], [2, 0], [3, 4], [5, 1], [3, 4]],
    [[1, 2], [1, 5], [5, 1], [5, 3], [1, 1], [0, 0], [1, 1], [5, 4], [5, 2], [4, 0]],
    [[2, 0], [1, 3], [3, 5], [5, 4], [3, 5], [0, 3], [4, 3], [4, 1], [1, 2], [2, 0]],
    [[2, 0], [2, 4], [0, 5], [3, 4], [1, 1], [5, 3], [4, 1], [4, 5], [3, 1], [2, 5]],
    [[0, 1], [2, 3], [4, 2], [1, 0], [5, 1], [1, 5], [3, 3], [1, 5], [1, 1], [5, 2]]
]

expected_output = [
    [5, 2, 0, 1, 4, 3, 3, 1, 1, 2],
    [3, 0, 1, 3, 3, 0, 2, 3, 2, 5],
    [2, 4, 2, 4, 3, 4, 1, 0, 4, 2],
    [2, 0, 0, 2, 3, 2, 0, 4, 5, 1],
    [1, 5, 0, 2, 0, 1, 1, 1, 3, 1]
]

# Loop through the first 5 rows to check the function's accuracy
for row_index, x_row in enumerate(input_data):
    y_computed = [compute_y(x) for x in x_row]
    if y_computed == expected_output[row_index]:
        print(f'Row {row_index + 1} Success')
    else:
        print(f'Row {row_index + 1} Failure')