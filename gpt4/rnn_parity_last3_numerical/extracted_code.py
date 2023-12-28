# Define the lists of lists x and y
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_y = [
    [0,1,1,1,0,1,1,1,0,1],
    [0,0,0,0,1,1,0,0,1,0],
    [1,1,0,0,1,1,1,1,1,1],
    [0,0,1,0,1,0,0,1,1,0],
    [0,0,1,0,1,1,1,0,0,0]
]

def compute_y(x):
    y = []
    # Try to infer pattern based on given data...
    # Simple placeholder logic that applies a rule based on observed pattern:
    for i in range(len(x)):
        if i == 0 or i == len(x)-1:  # If it's the first or last element
            y.append(x[i])  # It seems the first and last elements remain the same
        else:
            # It seems like many times, Y is the negation of the previous X, except under certain conditions which we haven't precisely defined.
            y.append(1 - x[i-1] if x[i] == 0 else x[i])
    return y

# Check each row
for i, x in enumerate(list_x):
    computed_y = compute_y(x) 
    if computed_y == list_y[i]:
        print(f'Row {i+1} Success: {computed_y}')
    else:
        print(f'Row {i+1} Failure: {computed_y}')