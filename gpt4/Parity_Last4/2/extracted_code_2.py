import itertools

# Define the rows of x and y provided
x_rows = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1],
]

y_rows = [
    [0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,0,1,1,1],
    [1,1,0,1,1,0,0,0,0,0],
    [0,0,1,0,1,1,1,0,1,1],
    [0,0,1,0,1,0,0,1,1,1],
]

# This function is a placeholder for the actual logic you may define to compute y from x
def compute_y(x_list):
    # We will use a heuristic: y[i] = not x[i] if i > 0 else x[i]
    # Essentially flipping every bit except the first one
    # This is an arbitrary choice and is used to demonstrate the structure of the solution
    return [x_list[i] if i == 0 else 1 - x_list[i] for i in range(len(x_list))]

# Check each row and determine if the transformation matches the given y
success = True
for i in range(5):
    computed_y = compute_y(x_rows[i])
    if computed_y != y_rows[i]:
        success = False
        print(f'Failure on row {i+1}')
        break

if success:
    print('Success')