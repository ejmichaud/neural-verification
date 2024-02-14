def compute_y(x):
    # This function is based on trying to find a pattern in the provided lists.
    # Since the assignment did not specify the exact logic, this function is a best guess approach.
    
    # Replace this with the logic you find that computes y from x.
    # As a placeholder, this simply flips 1s and 0s, which does not represent any real logic.
    return [1 - xi for xi in x]

# Input data from the table (x lists and y lists)
x_lists = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

y_lists = [
    [1,0,1,1,1,0,1,1,1,0],
    [1,1,1,1,0,1,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,1,0,1,1,1],
    [1,1,0,0,0,0,0,1,0,0]
]

# Flag for overall success or failure
overall_success = True

# Check the computed y against the expected y for each row
for i in range(len(x_lists)):
    computed_y = compute_y(x_lists[i])
    if computed_y == y_lists[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')
        overall_success = False

# Print overall success or failure
if overall_success:
    print('Success for all rows')
else:
    print('Failure: not all rows matched')