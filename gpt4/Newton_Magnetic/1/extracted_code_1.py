def compute_y(x_list):
    y_list = []
    for index, (a, b) in enumerate(x_list):
        if index == 0:  # The first element seems to be the same as x
            y_list.append([a, b])
        else:
            # Deduce a pattern from previous elements with trial and error, checking each combination
            prev_a, prev_b = y_list[index - 1]
            new_a = prev_a + a - index  # Tentative transformation for a
            new_b = prev_b + b + index  # Tentative transformation for b
            y_list.append([new_a, new_b])
    return y_list

# Test the compute_y function with first 5 rows
x = [
    [5, -8], [-6, 9], [8, 7], [9, -5], [-3, 4],
    [-9, 4], [-3, 7], [7, 9], [-2, 5], [6, 6],
    [4, 5], [9, 2], [-4, 1], [4, 6], [2, -5],
    [-5, -10], [-8, 9], [2, -8], [8, 6], [3, -3],
    [-10, -9], [10, -7], [-6, -8], [3, 2], [1, 9]
]

expected_y = [
    [5, -8], [12, -2], [21, 18], [19, 42], [-10, 68],
    [-9, 4], [-25, 6], [-36, 1], [-44, -10], [-35, -23],
    [4, 5], [12, 16], [5, 36], [-18, 55], [-58, 46],
    [-5, -10], [-8, -16], [-3, -33], [27, -39], [66, -18],
    [-10, -9], [-1, -35], [28, -60], [85, -54], [137, 18]
]

for i in range(5):  # Test for first 5 lists
    result = compute_y(x[i*5:i*5+5])
    print('Success' if result == expected_y[i*5:i*5+5] else 'Failure', result)