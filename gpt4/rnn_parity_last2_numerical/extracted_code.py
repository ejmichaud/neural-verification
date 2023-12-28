def compute_y(x):
    y = [0] * len(x)     # Initialize y with the same length as x, filled with zeros.
    i = 0                # Start index at 0
    while i < len(x):    # Iterate through each element of x
        if x[i] == 1:            # Check for '1' in 'x'
            # Check if next element is '0' or if it's the last element
            if i+1 == len(x) or x[i+1] == 0:
                y[i] = 1          # Keep '1' at the same position
                i += 1            # Move to the next element
            else:
                # Find the next '0' in y to turn into '1'
                j = (i + 1) % len(x)  # Start searching from the next position
                while x[j] != 0:      # Skip positions that are '1' in x
                    j = (j + 1) % len(x)
                y[j] = 1              # Flip the '0' to a '1' in y
                i += 2                # Skip the next element since it's involved in this transformation
        else:
            i += 1  # Move to the next element if '0' in 'x'
    return y

# Define the first 5 rows as provided
rows_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

# Define the corresponding rows of y as provided
rows_y = [
    [0,1,1,0,0,1,1,0,0,1],
    [0,0,0,0,1,1,1,0,0,1],
    [1,1,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,1,1,0,0],
    [0,0,1,0,0,0,0,1,1,0]
]

# Check if the algorithm works for each row
for i, x in enumerate(rows_x):
    computed_y = compute_y(x)
    if computed_y == rows_y[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")