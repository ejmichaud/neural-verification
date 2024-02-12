# Define the first 5 rows of lists x and their corresponding lists y
list_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77],
]

list_ys = [
    [42, 109, 185, 199, 183, 151, 95, 105, 129, 107],
    [78, 92, 102, 156, 109, 167, 172, 213, 249, 183],
    [49, 125, 198, 209, 259, 196, 164, 222, 210, 216],
    [72, 152, 227, 256, 217, 201, 165, 212, 211, 157],
    [86, 108, 185, 204, 125, 126, 92, 167, 253, 307],
]

# Function to compute list y from list x based on cumulative sum
def compute_cumulative_sum(list_x):
    cumulative_sum = []
    total = 0
    for x in list_x:
        total += x
        cumulative_sum.append(total)
    return cumulative_sum

# Iterate through each list x, compute list y, and check if it matches the provided list y
for i in range(5):
    calculated_y = compute_cumulative_sum(list_xs[i])
    # Verify if calculated list y matches the provided list y
    if calculated_y == list_ys[i]:
        print(f'Row {i+1} check: Success')
    else:
        print(f'Row {i+1} check: Failure')