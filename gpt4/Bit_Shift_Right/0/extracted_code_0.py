def shift_right_by_one(lst):
    return [lst[-1]] + lst[:-1]

# Given list of x and y pairs for the first 5 rows
list_pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,0,1,0,0,0,1,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,1,0,1,1,1]),
    ([1,0,1,1,1,1,1,1,1,1], [0,1,0,1,1,1,1,1,1,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,1,1,0,1,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,1,1,1,1,0,1])
]

# Process each pair of x and y lists and check if the output matches
for x_list, expected_y_list in list_pairs:
    computed_y_list = shift_right_by_one(x_list)
    if computed_y_list == expected_y_list:
        print('Success')
    else:
        print('Failure')

        # Print debug information
        print(f'Computed y: {computed_y_list}')
        print(f'Expected y: {expected_y_list}')