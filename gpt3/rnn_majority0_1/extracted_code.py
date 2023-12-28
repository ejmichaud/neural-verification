def compute_y_from_x(x):
    y = [0 if val == 0 else 1 for val in x]
    return y

# Input data
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

# Expected list y
list_y_expected = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1]
]

# Compute and check list y for the first 5 rows
for i in range(5):
    y_computed = compute_y_from_x(list_x[i])
    if y_computed == list_y_expected[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")