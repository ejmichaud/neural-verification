# List of lists x and expected lists y
data = [
    ([2, 7, 6, 4, 6, 5, 0, 4, 0, 3], [2, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
    ([8, 4, 0, 4, 1, 2, 5, 5, 7, 6], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
    ([9, 6, 3, 1, 9, 3, 1, 9, 7, 9], [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    ([2, 0, 5, 9, 3, 4, 9, 6, 2, 0], [2, 2, 5, 9, 9, 9, 9, 9, 9, 9]),
    ([6, 2, 7, 9, 7, 3, 3, 4, 3, 7], [6, 6, 7, 9, 9, 9, 9, 9, 9, 9]),
]

# Function to compute list y from list x
def compute_y_from_x(list_x):
    list_y = [list_x[0]]
    for i in range(1, len(list_x)):
        list_y.append(max(list_y[i - 1], list_x[i]))
    return list_y

# Process pairs of list x and list y and check for matches
for i, (list_x, expected_list_y) in enumerate(data):
    # Compute list y from list x
    computed_list_y = compute_y_from_x(list_x)
    
    # Check if the computed list y matches the expected list y
    if computed_list_y == expected_list_y:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')