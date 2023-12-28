def shift_and_pad(x):
    # Shift x to the right by 3 positions and pad with zeroes
    return [0, 0, 0] + x[:-3]

# Define the first five rows of list x and y from the table
list_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

list_ys = [
    [0, 0, 0, 42, 67, 76, 14, 26, 35, 20],
    [0, 0, 0, 78, 14, 10, 54, 31, 72, 15],
    [0, 0, 0, 49, 76, 73, 11, 99, 13, 41],
    [0, 0, 0, 72, 80, 75, 29, 33, 64, 39],
    [0, 0, 0, 86, 22, 77, 19, 7, 23, 43]
]

# Compute and check each row for a match
for i in range(len(list_xs)):
    calculated_y = shift_and_pad(list_xs[i])
    if calculated_y == list_ys[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")