def compute_y(x):
    cumulative_sum = 0
    y = []
    for num in x:
        cumulative_sum += num   # Calculate cumulative sum
        y.append(cumulative_sum) # Append to list y
    return y

# Define the input and expected output for the first 5 rows
input_lists = [
    [5, -8, -6, 9, 8, 7, 9, -5, -3, 4],
    [-3, 9, 10, 8, 5, 6, 10, 3, -1, 0],
    [-9, 4, -3, 7, 7, 9, -2, 5, 6, 6],
    [-1, 2, -3, -3, -5, -3, 1, 7, 3, 5],
    [4, 5, 9, 2, -4, 1, 4, 6, 2, -5]
]

output_lists = [
    [4, -1, -13, -17, -14, -5, 12, 23, 30, 40],
    [-4, 0, 13, 33, 57, 86, 124, 164, 202, 239],
    [-10, -17, -28, -33, -32, -23, -17, -7, 8, 28],
    [-2, -3, -8, -17, -32, -51, -70, -83, -94, -101],
    [3, 10, 25, 41, 52, 63, 77, 96, 116, 130]
]

# Test the function on the first 5 rows and check the output
for i in range(5):
    computed_y = compute_y(input_lists[i])
    if computed_y == output_lists[i]:
        print(f'Row {i+1}: Success')
    else:
        print(f'Row {i+1}: Failure')
        print(f'Computed list y: {computed_y}')
        print(f'Expected list y: {output_lists[i]}')