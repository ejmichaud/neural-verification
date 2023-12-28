def compute_y(x):
    y = x.copy()  # Start with a copy of x
    for i in range(len(x)-1):
        for j in range(i+1, len(x)):
            if x[j] >= y[i]:  # Ensure that we use the newly updated y value for comparison
                y[i] = x[j]
            else:
                # Once we find an element that is not greater than or equal to the current,
                # we stop processing further for the current i and move to the next element.
                break
    return y

# Define the first 5 rows of x and y
rows_x = [
    [2,2,1,4,1,0,0,4,0,3],
    [3,4,0,4,1,2,0,0,2,1],
    [4,1,3,1,4,3,1,4,2,4],
    [2,0,0,4,3,4,4,1,2,0],
    [1,2,2,4,2,3,3,4,3,2]
]

rows_y = [
    [2,2,2,2,1,1,0,0,0,0],
    [3,3,0,4,4,4,0,0,0,0],
    [4,1,1,1,1,1,1,1,1,4],
    [2,0,0,0,0,0,4,4,4,0],
    [1,1,2,2,2,2,2,2,2,2]
]

# Check for each row
for i in range(5):
    computed_y = compute_y(rows_x[i])
    if computed_y == rows_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure: Expected {rows_y[i]} but got {computed_y}")