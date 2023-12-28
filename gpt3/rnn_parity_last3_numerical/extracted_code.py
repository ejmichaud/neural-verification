def compute_y(x):
    y = []
    for i in range(len(x)):
        if i == 0:
            y.append(x[i])
        else:
            y.append(x[i] ^ x[i-1])
    return y

# Input data
x_list = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]
y_list = [
    [0,1,1,1,0,1,1,1,0,1],
    [0,0,0,0,1,1,0,0,1,0],
    [1,1,0,0,1,1,1,1,1,1],
    [0,0,1,0,1,0,0,1,1,0],
    [0,0,1,0,1,1,1,0,0,0]
]

# Computing and checking results
for i in range(5):
    y_computed = compute_y(x_list[i])
    if y_computed == y_list[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")