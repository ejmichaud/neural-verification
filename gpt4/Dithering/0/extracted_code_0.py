# Define the simple function to compute y based on the observation
def compute_y(x):
    # Here, we are using the crude thresholding rule identified earlier
    return [1 if val > 6 else 0 for val in x]

# Define your input data (first 5 rows of list x)
list_x_data = [
    [6, 3, 12, 14, 10, 7, 12, 4, 6, 9],
    [2, 6, 10, 10, 7, 4, 3, 7, 7, 2],
    [5, 4, 1, 7, 11, 13, 5, 1, 15, 11],
    [4, 0, 11, 9, 5, 12, 11, 8, 0, 10],
    [10, 14, 9, 15, 11, 11, 15, 14, 13, 13]
]

# Define your expected output data (first 5 rows of list y)
expected_y_data = [
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
]

# Use the function to compute the y values and compare them to the expected
for i in range(len(list_x_data)):
    computed_y = compute_y(list_x_data[i])
    # Compare computed results with expected results
    if computed_y == expected_y_data[i]:
        print(f"Success on row {i+1}: Computed y matches expected y.")
    else:
        print(f"Failure on row {i+1}: Computed y does not match expected y.")