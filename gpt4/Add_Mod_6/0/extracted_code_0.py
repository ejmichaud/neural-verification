def compute_y(x):
    # Placeholder for the function, need to determine the correct operation
    # Assuming a simple operation (-1) as a starting point
    return [xi - 1 for xi in x]

rows_x = [
    [0, 5, 4, 4, 0, 5, 4, 2, 4, 5],
    [4, 4, 2, 0, 3, 4, 5, 1, 3, 4],
    [1, 2, 1, 5, 5, 1, 5, 3, 1, 1],
    [0, 0, 1, 1, 5, 4, 5, 2, 4, 0],
    [2, 0, 1, 3, 3, 5, 5, 4, 3, 5]
]

rows_y = [
    [0, 5, 3, 1, 1, 0, 4, 0, 4, 3],
    [4, 2, 4, 4, 1, 5, 4, 5, 2, 0],
    [1, 3, 4, 3, 2, 3, 2, 5, 0, 1],
    [0, 0, 1, 2, 1, 5, 4, 0, 4, 4],
    [2, 2, 3, 0, 3, 2, 1, 5, 2, 1]
]

success = True
for i in range(5):
    calculated_y = compute_y(rows_x[i])
    if calculated_y != rows_y[i]:
        print(f"Failure on row {i+1}")
        success = False
        break  # Break out of the loop on the first mismatch

if success:
    print("Success")