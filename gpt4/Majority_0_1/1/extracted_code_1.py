def compute_y(x):
    y = [0] * len(x)  # Start with a list of zeros the same length as x
    for i in range(2, len(x)):
        if x[i] == 1 and x[i-1] == 1 and x[i-2] == 1 and (i+1 < len(x) and x[i+1] == 1):
            y[i] = 1
    return y

# Define the lists
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_y = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,1,1,1,1,1,1]
]

# Check each list
for i in range(len(list_x)):
    output_y = compute_y(list_x[i])
    if output_y == list_y[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")