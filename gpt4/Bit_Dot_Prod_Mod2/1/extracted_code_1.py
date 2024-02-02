# Function to compute y using a hypothetical rule
def compute_y(x_pair):
    # Hypothesis: y = x[0] OR (NOT x[1])
    return int(x_pair[0] or (not x_pair[1]))

# Define list of x pairs (first 5 rows)
list_x = [
    [0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0],
    [1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0],
    [0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0],
    [0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1],
    [0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0],
]

# Define expected y values (first 5 rows)
expected_y = [
    [0,0,0,0,0,0,0,0,1,1],
    [0,1,0,1,0,0,1,1,1,1],
    [0,1,0,0,1,1,1,1,1,1],
    [0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,1,0,1,0,1,1],
]

# Compute y for each list of x pairs and check against expected y
for row_idx, (x_row, y_expected) in enumerate(zip(list_x, expected_y)):
    y_computed = [compute_y(x_pair) for x_pair in x_row]
    if y_computed == y_expected:
        print(f"Success for row {row_idx+1}")
    else:
        print(f"Failure for row {row_idx+1}")