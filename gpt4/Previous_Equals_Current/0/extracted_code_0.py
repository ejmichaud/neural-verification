# Hypothesized function to compute y[i] based on x
def compute_y(x, i):
    if i > 0 and i < len(x) - 1 and x[i - 1] == x[i + 1]:
        return 1
    else:
        return 0

# Input data: first 5 rows of list x and list y
rows_x = [
    [0,7,7,4,7,7,2,2,6,0],
    [0,6,7,5,4,4,2,6,3,0],
    [6,7,1,5,3,3,1,6,3,3],
    [3,6,0,6,5,6,4,7,6,6],
    [6,0,3,5,3,3,1,3,1,3],
]

rows_y = [
    [1,0,1,0,0,1,0,1,0,0],
    [1,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0,0,0,0],
]

# Process each row and check if the computed y matches the given y
for i, x in enumerate(rows_x):
    # Compute y for current row
    computed_y = [compute_y(x, j) for j in range(len(x))]
    
    # Check if the computed y matches the expected y
    if computed_y == rows_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')

        # Printing the mismatching lists for diagnostics
        print(f'Expected y: {rows_y[i]}')
        print(f'Computed y: {computed_y}')