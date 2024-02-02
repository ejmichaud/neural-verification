def compute_y(x_list):
    y = []
    for i, x in enumerate(x_list):
        if i == 0:
            y.append(x[0])
        else:
            y.append(y[-1] + x[0] * i)
    return y

# Provided lists x and y
input_rows_x = [
    [5, -8, -6, 9, 8, 7, 9, -5, -3, 4],
    [-3, 9, 10, 8, 5, 6, 10, 3, -1, 0],
    [-9, 4, -3, 7, 7, 9, -2, 5, 6, 6],
    [-1, 2, -3, -3, -5, -3, 1, 7, 3, 5],
    [4, 5, 9, 2, -4, 1, 4, 6, 2, -5]
]

output_rows_y = [
    [5, 2, -7, -7, 1, 16, 40, 59, 75, 95],
    [-3, 3, 19, 43, 72, 107, 152, 200, 247, 294],
    [-9, -14, -22, -23, -17, -2, 11, 29, 53, 83],
    [-1, 0, -2, -7, -17, -30, -42, -47, -49, -46],
    [4, 13, 31, 51, 67, 84, 105, 132, 161, 185]
]

# Check each row
for index, row_x in enumerate(input_rows_x):
    computed_y = compute_y([[x] for x in row_x])
    given_y = output_rows_y[index]
    if computed_y == given_y:
        print(f"Row {index + 1}: Success")
    else:
        print(f"Row {index + 1}: Failure")
        print(f"Expected: {given_y}")
        print(f"Computed: {computed_y}")