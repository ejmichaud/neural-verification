def compute_y(x):
    y = []
    for i in x:
        if i < 50:
            y.append(0)
        else:
            y.append(1)
    return y

# Input lists x and y
x_lists = [
    [42,32,32,14,26,35,20,24,50,13],
    [78,14,1,1,31,1,15,95,67,6],
    [49,1,1,11,99,13,41,1,87,1],
    [72,80,75,29,32,32,39,76,32,32],
    [86,4,4,4,4,23,43,94,93,4]
]
y_lists = [
    [0,1,1,0,0,0,0,0,0,0],
    [0,0,1,1,0,1,0,0,0,0],
    [0,1,1,0,0,0,0,1,0,1],
    [0,0,0,0,1,1,0,0,1,1],
    [0,1,1,1,1,0,0,0,0,1]
]

# Compute list y and check if it matches the actual list y
for i in range(5):
    computed_y = compute_y(x_lists[i])
    if computed_y == y_lists[i]:
        print("Success")
    else:
        print("Failure")