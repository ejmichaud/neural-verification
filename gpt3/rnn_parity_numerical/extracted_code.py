def compute_y(x):
    y = [0]*len(x)
    for i in range(len(x)-1):
        if x[i] == 0 and x[i+1] == 1:
            y[i] = 1
            y[i+1] = 0
        else:
            y[i] = x[i]
    y[-1] = x[-1]  # Set the last element of y to be the same as the last element of x
    return y

# Given lists x and y
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_y_expected = [
    [0,1,1,1,1,0,0,0,0,1],
    [0,0,0,0,1,1,0,1,0,0],
    [1,1,0,1,0,1,0,1,0,1],
    [0,0,1,0,1,1,0,0,0,0],
    [0,0,1,0,1,0,1,1,0,1]
]

# Compute list y and check if it matches list_y_expected
for i in range(5):
    y_actual = compute_y(list_x[i])
    if y_actual == list_y_expected[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")