import numpy as np

# Data from the table for the first 5 rows
list_x = [
    [6, 3, 12, 14, 10, 7, 12, 4, 6, 9],
    [2, 6, 10, 10, 7, 4, 3, 7, 7, 2],
    [5, 4, 1, 7, 11, 13, 5, 1, 15, 11],
    [4, 0, 11, 9, 5, 12, 11, 8, 0, 10],
    [10, 14, 9, 15, 11, 11, 15, 14, 13, 13]
]

actual_list_y = [
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
]

def compute_y_based_on_mean(list_x):
    # This function would compare each element with the mean of the list and create list_y
    computed_list_y = []
    
    for row in list_x:
        # Calculate mean of the current row
        mean_value = np.mean(row)
        
        # Function to check if the element is greater than the mean
        computed_row_y = [1 if x > mean_value else 0 for x in row]
        
        computed_list_y.append(computed_row_y)
        
    return computed_list_y

# Compute list y using the function
computed_list_y = compute_y_based_on_mean(list_x)

# Verify if the computed list y matches the actual list y
for i, (computed_y, actual_y) in enumerate(zip(computed_list_y, actual_list_y)):
    if computed_y == actual_y:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')