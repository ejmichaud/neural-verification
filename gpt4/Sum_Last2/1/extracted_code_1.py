# Define the list of lists x
list_of_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77],
]

# Define the correct list of lists y
correct_list_of_ys = [
    [42, 109, 143, 90, 40, 61, 55, 44, 74, 63],
    [78, 92, 24, 64, 85, 103, 87, 110, 162, 73],
    [49, 125, 149, 84, 110, 112, 54, 110, 156, 106],
    [72, 152, 155, 104, 62, 97, 103, 115, 108, 42],
    [86, 108, 99, 96, 26, 30, 66, 137, 187, 170],
]

def compute_cumulative_sum(x_list):
    y_list = []
    sum_so_far = 0
    for x in x_list:
        sum_so_far += x
        y_list.append(sum_so_far)
    return y_list

# Iterate over the lists and compute y from x, then check if it matches the given y
for i in range(len(list_of_xs)):
    computed_y = compute_cumulative_sum(list_of_xs[i])
    if computed_y == correct_list_of_ys[i]:
        print(f'Row {i+1}: Success')
    else:
        print(f'Row {i+1}: Failure')

# Note: The row numbers in prints are 1-indexed for readability, matching human typical counting.