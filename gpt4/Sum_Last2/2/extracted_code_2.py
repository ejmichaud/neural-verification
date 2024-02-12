def calculate_y(x):
    y = []
    cumulative_sum = 0
    for value in x:
        cumulative_sum += value
        if cumulative_sum > 100:
            cumulative_sum = value
        y.append(cumulative_sum)
    return y

# Define the first five rows of list x from the question
list_of_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77],
]

# Define the corresponding y lists from the question
given_ys = [
    [42, 109, 143, 90, 40, 61, 55, 44, 74, 63],
    [78, 92, 24, 64, 85, 103, 87, 110, 162, 73],
    [49, 125, 149, 84, 110, 112, 54, 110, 156, 106],
    [72, 152, 155, 104, 62, 97, 103, 115, 108, 42],
    [86, 108, 99, 96, 26, 30, 66, 137, 187, 170],
]

# Iterate through each row, calculate y, and check if it matches the given y
for i, x in enumerate(list_of_xs):
    calculated_y = calculate_y(x)
    if calculated_y == given_ys[i]:
        print(f"Row {i + 1} - Success")
    else:
        print(f"Row {i + 1} - Failure")