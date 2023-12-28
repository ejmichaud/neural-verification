def compute_cumulative_sum(x):
    # Initialize an empty list to store cumulative sums
    cumulative_sum = []
    sum_so_far = 0
    for value in x:
        sum_so_far += value
        cumulative_sum.append(sum_so_far)
    return cumulative_sum

# Define the first 5 rows of x and the corresponding y
list_of_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

list_of_y = [
    [42, 109, 185, 199, 225, 260, 280, 304, 354, 367],
    [78, 92, 102, 156, 187, 259, 274, 369, 436, 442],
    [49, 125, 198, 209, 308, 321, 362, 431, 518, 537],
    [72, 152, 227, 256, 289, 353, 392, 468, 500, 510],
    [86, 108, 185, 204, 211, 234, 277, 371, 464, 541]
]

# Now, compute y from x using the defined function and check the results
for i in range(len(list_of_x)):
    x = list_of_x[i]
    y_computed = compute_cumulative_sum(x)
    y_provided = list_of_y[i]

    # Check if the computed y matches the provided y
    if y_computed == y_provided:
        print('Success')
    else:
        print('Failure')