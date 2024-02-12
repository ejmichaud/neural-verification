# Define the first 5 rows of list x and corresponding list y
x_rows = [
    [[2, 2],[1, 4],[1, 0],[0, 4],[0, 3],[3, 4],[0, 4],[1, 2],[0, 0],[2, 1]],
    [[4, 1],[3, 1],[4, 3],[1, 4],[2, 4],[2, 0],[0, 4],[3, 4],[4, 1],[2, 0]],
    [[1, 2],[2, 4],[2, 3],[3, 4],[3, 2],[0, 4],[0, 4],[1, 4],[0, 4],[3, 3]],
    [[1, 0],[0, 0],[0, 1],[3, 0],[1, 1],[2, 4],[4, 3],[3, 4],[3, 2],[3, 1]],
    [[4, 1],[1, 3],[2, 0],[4, 3],[0, 3],[2, 2],[0, 4],[1, 0],[1, 4],[1, 4]],
]

y_rows = [
    [4,0,2,4,3,2,0,4,0,3],
    [0,0,3,1,2,3,4,2,1,3],
    [3,1,1,3,1,0,0,1,0,2],
    [1,0,1,3,2,1,3,3,1,0],
    [0,0,3,2,4,4,4,1,0,1],
]

# Placeholder function to calculate y from x
# We will try to implement a function based on the given pattern or relationship
def calculate_y(x_pair):
    a, b = x_pair
    return 0  # This needs to be replaced by the actual relation/pattern

# Check the first 5 rows
for i in range(5):
    computed_y = [calculate_y(pair) for pair in x_rows[i]]  # Compute y for each pair in x
    if computed_y == y_rows[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure. Expected {y_rows[i]}, got {computed_y}')

# The above code will print 'Failure' for each row as the calculate_y function does not have the correct pattern yet
# Implement the correct calculate_y function based on the pattern found in the data