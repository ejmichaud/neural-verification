def compute_y(x):
    y = [0]  # Start with a '0' since y[0] is always '0'
    for i in range(1, len(x)):
        # if current and previous `x` are `1`, set `y` to `1`, else `0`
        y.append(1 if (x[i] == 1 and x[i-1] == 1) else 0)
    return y

# Given rows of x and y to test the function
rows_x = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
]

rows_y = [
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 1]
]

# Testing the function on the provided rows and printing the result
for idx, x in enumerate(rows_x):
    calculated_y = compute_y(x)
    if calculated_y == rows_y[idx]:
        print(f"Row {idx + 1} Success")
    else:
        print(f"Row {idx + 1} Failure")