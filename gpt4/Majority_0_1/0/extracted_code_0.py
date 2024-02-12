def compute_y(x):
    y = [0] * len(x) # Initialize y with zeros
    for i in range(2, len(x)):
        if x[i] == 1 and x[i-1] == 1 and x[i-2] == 1:
            y[i] = 1
    return y

# Given pairs of lists x and y
list_of_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_of_y = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1]
]

# Checking the results
for i in range(len(list_of_x)):
    calculated_y = compute_y(list_of_x[i])
    if calculated_y == list_of_y[i]:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')

# For better output clarity, let's also print out the results
        print(f'Expected y : {list_of_y[i]}')
        print(f'Calculated y: {calculated_y}')