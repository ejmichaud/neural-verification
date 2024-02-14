# Function to (theoretically) compute y from x, placeholder since we don't have the actual formula
def compute_y(x_list):
    # There's no given formula or pattern, so the output below is simply a placeholder.
    # This would normally be replaced with actual logic to compute y.
    return [0] * len(x_list)

# Define the first five rows of inputs (list x) and expected outputs (list y)
xs = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]],
]

ys = [
    [1,0,1,0,1,0,0,1,0,0],
    [1,0,1,1,1,1,0,0,0,1],
    [0,0,1,0,1,0,0,0,0,1],
    [0,0,1,1,0,1,0,0,1,0],
    [1,1,0,1,0,1,1,1,1,0],
]

# Check if the computed ys match the actual ys for the first five rows
for i in range(5):
    predicted_y = compute_y(xs[i])
    actual_y = ys[i]
    if predicted_y == actual_y:
        print(f'Row {i+1} Success')  # i+1 for one-indexed row number
    else:
        print(f'Row {i+1} Failure')

# Since we don't have the actual mapping, we expect to see 'Failure' for all rows