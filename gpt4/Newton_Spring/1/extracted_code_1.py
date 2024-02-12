def cumulative_sum(x_list):
    y_list = []
    sum_so_far = 0
    for x in x_list:
        sum_so_far += x[0]  # Assumes x consists of lists with one element each
        y_list.append([sum_so_far])
    return y_list

# Define the first 5 rows of list x and list y
input_x = [
    [[5], [-8], [-6], [9], [8], [7], [9], [-5], [-3], [4]],
    [[-3], [9], [10], [8], [5], [6], [10], [3], [-1], [0]],
    [[-9], [4], [-3], [7], [7], [9], [-2], [5], [6], [6]],
    [[-1], [2], [-3], [-3], [-5], [-3], [1], [7], [3], [5]],
    [[4], [5], [9], [2], [-4], [1], [4], [6], [2], [-5]]
]

expected_y = [
    [[5], [-3], [-14], [-2], [20], [29], [18], [-16], [-37], [-17]],
    [[-3], [6], [19], [21], [7], [-8], [-5], [6], [10], [4]],
    [[-9], [-5], [1], [13], [19], [15], [-6], [-16], [-4], [18]],
    [[-1], [1], [-1], [-5], [-9], [-7], [3], [17], [17], [5]],
    [[4], [9], [14], [7], [-11], [-17], [-2], [21], [25], [-1]]
]

# Process each row, compute list y, and check against the expected list y
for i in range(len(input_x)):
    computed_y = cumulative_sum(input_x[i])
    if computed_y == expected_y[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")
        print(f"Expected: {expected_y[i]}, Computed: {computed_y}")