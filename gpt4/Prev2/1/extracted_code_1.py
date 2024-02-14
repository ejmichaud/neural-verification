# Function to transform list x to y
def transform_list_x_to_y(x):
    return [0, 0] + x[:-2]

# Given lists x and y
pairs = [
    ([42, 67, 76, 14, 26, 35, 20, 24, 50, 13], [0, 0, 42, 67, 76, 14, 26, 35, 20, 24]),
    ([78, 14, 10, 54, 31, 72, 15, 95, 67, 6], [0, 0, 78, 14, 10, 54, 31, 72, 15, 95]),
    ([49, 76, 73, 11, 99, 13, 41, 69, 87, 19], [0, 0, 49, 76, 73, 11, 99, 13, 41, 69]),
    ([72, 80, 75, 29, 33, 64, 39, 76, 32, 10], [0, 0, 72, 80, 75, 29, 33, 64, 39, 76]),
    ([86, 22, 77, 19, 7, 23, 43, 94, 93, 77], [0, 0, 86, 22, 77, 19, 7, 23, 43, 94]),
]

# Iterate over the first five pairs
for x, expected_y in pairs[:5]:
    # Compute y from x
    y = transform_list_x_to_y(x)
    # Check if computed y matches the expected y and print the result
    if y == expected_y:
        print('Success')
    else:
        print('Failure')

# It should print 'Success' five times