def compute_y(x):
    y = []
    for i in range(len(x)):
        if x[i] == 3 and (i + 1) < len(x) and x[i + 1] == 0:
            y.append(0)
        elif x[i] == 3:
            y.append(3)
        elif x[i] == 2:
            y.append(2)
        else:
            y.append(0)
    return y

# Given lists x
lists_x = [
    [2,3,0,2,2,3,0,0,2,1],
    [2,2,2,2,3,0,3,3,3,2],
    [1,0,1,3,3,1,1,1,3,3],
    [0,0,3,1,1,0,3,0,0,2],
    [2,2,1,3,3,3,3,2,1,1]
]

# Corresponding lists y, for comparison
lists_y = [
    [2,2,0,2,2,2,2,0,2,2],
    [2,2,2,2,2,2,2,2,2,2],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [2,2,2,2,2,3,3,3,3,3]
]

# Check if the output matches list y and print 'Success' or 'Failure'
for idx, x in enumerate(lists_x):
    y_computed = compute_y(x)
    if y_computed == lists_y[idx]:
        print(f'Row {idx+1} Success')
    else:
        print(f'Row {idx+1} Failure, expected {lists_y[idx]}, but got {y_computed}')