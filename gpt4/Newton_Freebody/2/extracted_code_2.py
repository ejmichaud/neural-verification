# Given input lists of x and y
input_lists_x = [
    [5, -8, -6, 9, 8, 7, 9, -5, -3, 4],
    [-3, 9, 10, 8, 5, 6, 10, 3, -1, 0],
    [-9, 4, -3, 7, 7, 9, -2, 5, 6, 6],
    [-1, 2, -3, -3, -5, -3, 1, 7, 3, 5],
    [4, 5, 9, 2, -4, 1, 4, 6, 2, -5]
]

input_lists_y = [
    [5, 2, -7, -7, 1, 16, 40, 59, 75, 95],
    [-3, 3, 19, 43, 72, 107, 152, 200, 247, 294],
    [-9, -14, -22, -23, -17, -2, 11, 29, 53, 83],
    [-1, 0, -2, -7, -17, -30, -42, -47, -49, -46],
    [4, 13, 31, 51, 67, 84, 105, 132, 161, 185]
]

# Function to compute y from x using prefix sum
def compute_y_from_x(list_x):
    running_total = 0
    computed_y = []
    for x in list_x:
        running_total += x  # Add current x to the running total
        computed_y.append(running_total)  # Append the running total to y
    return computed_y

# Check the computed y against the input y for the first 5 rows
for i in range(5):  # Only iterate through the first 5 rows
    computed_y = compute_y_from_x(input_lists_x[i])
    if computed_y == input_lists_y[i]:
        print(f'Success for row {i+1}')
    else:
        print(f'Failure for row {i+1}')