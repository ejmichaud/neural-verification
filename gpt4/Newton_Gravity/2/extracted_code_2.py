# Define the input lists x and the expected lists y for the first 5 rows
input_output_pairs = [
    ([[5], [-8], [-6], [9], [8], [7], [9], [-5], [-3], [4]], [4, -1, -13, -17, -14, -5, 12, 23, 30, 40]),
    ([[-3], [9], [10], [8], [5], [6], [10], [3], [-1], [0]], [-4, 0, 13, 33, 57, 86, 124, 164, 202, 239]),
    ([[-9], [4], [-3], [7], [7], [9], [-2], [5], [6], [6]], [-10, -17, -28, -33, -32, -23, -17, -7, 8, 28]),
    ([[-1], [2], [-3], [-3], [-5], [-3], [1], [7], [3], [5]], [-2, -3, -8, -17, -32, -51, -70, -83, -94, -101]),
    ([[4], [5], [9], [2], [-4], [1], [4], [6], [2], [-5]], [3, 10, 25, 41, 52, 63, 77, 96, 116, 130])
]

def compute_y_from_x(list_x):
    list_y_computed = [list_x[0][0] - 1]
    for i in range(1, len(list_x)):
        list_y_computed.append(list_y_computed[i - 1] + list_x[i][0])
    return list_y_computed

# Check each pair
for (x, expected_y) in input_output_pairs:
    computed_y = compute_y_from_x(x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure', computed_y, expected_y)