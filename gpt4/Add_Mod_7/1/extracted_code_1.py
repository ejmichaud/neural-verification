def compute_y(x_list):
    # Check if the list is empty or contains only one element
    if not x_list or len(x_list) == 1:
        return x_list

    result = []
    # Iterate over the list starting from the second element
    for i in range(len(x_list)):
        # Take the current element and add the previous element in x_list
        # Use modulo by 7 (since numbers seems to be between 0 and 6) to calculate the next element of y
        result.append((x_list[i] + (x_list[i-1] if i > 0 else 0)) % 7)
        
    return result

# Define the input lists x and expected output lists y
x_lists = [
    [1, 2, 4, 5, 4, 3, 5, 5, 0, 0],
    [0, 5, 6, 4, 1, 2, 6, 6, 2, 3],
    [1, 0, 0, 3, 3, 5, 1, 1, 2, 2],
    [2, 5, 0, 0, 5, 0, 4, 3, 6, 1],
    [0, 1, 5, 5, 6, 4, 0, 2, 5, 5]
]

y_lists = [
    [1, 3, 0, 5, 2, 5, 3, 1, 1, 1],
    [0, 5, 4, 1, 2, 4, 3, 2, 4, 0],
    [1, 1, 1, 4, 0, 5, 6, 0, 2, 4],
    [2, 0, 0, 0, 5, 5, 2, 5, 4, 5],
    [0, 1, 6, 4, 3, 0, 0, 2, 0, 5]
]

# Process each list x and check if the computed list y matches the expected list y
for x, expected_y in zip(x_lists, y_lists):
    computed_y = compute_y(x)
    print('Success' if computed_y == expected_y else 'Failure', f'expected: {expected_y}, computed: {computed_y}')