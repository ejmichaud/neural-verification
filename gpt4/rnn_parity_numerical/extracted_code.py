def compute_y_from_x(x):
    y = []
    for i in range(len(x)):
        if x[i] == 1:
            # At the borders, we treat the non-existing neighbors as if they are 0.
            left_neighbor = x[i-1] if i > 0 else 0
            right_neighbor = x[i+1] if i < len(x)-1 else 0
            if left_neighbor == right_neighbor == 0:
                y.append(0)
            else:
                y.append(1)
        else:  # x[i] == 0
            if i > 0 and x[i-1] == 1 and (i == len(x)-1 or x[i+1] == 0):
                y.append(1)
            else:
                y.append(0)
    return y

# Define the first five rows of lists x and y
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_y = [
    [0,1,1,1,1,0,0,0,0,1],
    [0,0,0,0,1,1,0,1,0,0],
    [1,1,0,1,0,1,0,1,0,1],
    [0,0,1,0,1,1,0,0,0,0],
    [0,0,1,0,1,0,1,1,0,1]
]

# Check each row
for i in range(5):
    calculated_y = compute_y_from_x(list_x[i])
    if calculated_y == list_y[i]:
        print(f'Success for row {i+1}')
    else:
        print(f'Failure for row {i+1}')

# The below code is for debugging purposes, to see the actual output vs. expected
#     print(f'Row {i+1} output:   ', calculated_y)
#     print(f'Row {i+1} expected: ', list_y[i])