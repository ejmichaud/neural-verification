# Define the list of x values (first five rows)
list_of_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Define the correct list of y values (first five rows)
list_of_ys = [
    [0, 0, 42, 67, 76, 14, 26, 35, 20, 24],
    [0, 0, 78, 14, 10, 54, 31, 72, 15, 95],
    [0, 0, 49, 76, 73, 11, 99, 13, 41, 69],
    [0, 0, 72, 80, 75, 29, 33, 64, 39, 76],
    [0, 0, 86, 22, 77, 19, 7, 23, 43, 94]
]

# Function to transform x into y
def compute_y_from_x(x_list):
    # Transform list x into list y by shifting and padding with zeros
    return [0, 0] + x_list[:-2]

# Check each row to see if the transformation matches the provided y
for i in range(5):
    computed_y = compute_y_from_x(list_of_xs[i])
    if computed_y == list_of_ys[i]:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')