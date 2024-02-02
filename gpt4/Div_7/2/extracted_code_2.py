def compute_y(x_list):
    y_list = []
    # Using a sliding window approach to sum every 3 consecutive elements and check if the sum is greater than 1.
    for i in range(len(x_list)):
        if i < 2:  # We do not have enough elements yet for a sum
            y_list.append(0)
        else:
            # Calculate the sum of the current element and the two preceding ones.
            if sum(x_list[i-2:i+1]) > 1:
                y_list.append(1)
            else:
                y_list.append(0)
    return y_list

# List of x and y to check against
x_data = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
]

y_data = [
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
]

# Checking the first 5 rows
for i in range(5):
    computed_y = compute_y(x_data[i])
    if computed_y == y_data[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")