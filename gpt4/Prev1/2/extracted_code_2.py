# Define the first 5 rows of list x and list y
list_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

list_y = [
    [0, 42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [0, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [0, 49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [0, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [0, 86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Process each row to compute y and check against the provided y
for i in range(len(list_x)):
    computed_y = [0] + list_x[i]
    if computed_y == list_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')