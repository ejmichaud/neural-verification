import numpy as np

# Define the first 5 input lists of x
input_lists = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77],
]

# Define the first 5 output lists of y
output_lists = [
    [42, 109, 143, 90, 40, 61, 55, 44, 74, 63],
    [78, 92, 24, 64, 85, 103, 87, 110, 162, 73],
    [49, 125, 149, 84, 110, 112, 54, 110, 156, 106],
    [72, 152, 155, 104, 62, 97, 103, 115, 108, 42],
    [86, 108, 99, 96, 26, 30, 66, 137, 187, 170],
]

# Function to compute list y from list x
def compute_y_from_x(x_list):
    return np.cumsum(x_list).tolist()

# Check each row
for i in range(5):
    # Compute y from x
    computed_y = compute_y_from_x(input_lists[i])
    
    # Check if the computed y matches the given y
    if computed_y == output_lists[i]:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')