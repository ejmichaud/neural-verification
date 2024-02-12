def compute_y(x_list):
    y_list = []
    for i in range(len(x_list)):
        if x_list[i] == [1, 1]:
            if i == 0 or x_list[i-1] == [0, 0] or x_list[i-1] == [0, 1]:
                y_list.append(1)
            else:
                y_list.append(0)
        else:
            y_list.append(0)
    return y_list

# Define the first 5 rows of list x and y as per your input
x_data = [
    [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]],
    [[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 0]],
    [[0, 0],[1, 1],[1, 1],[1, 0],[1, 1],[0, 1],[0, 1],[0, 1],[1, 0],[0, 0]],
    [[0, 0],[0, 0],[0, 1],[1, 0],[1, 1],[1, 1],[0, 1],[0, 1],[1, 1],[0, 1]],
    [[0, 1],[0, 1],[0, 0],[1, 0],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 0]]
]

y_data = [
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,0,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,0],
    [0,0,0,0,1,1,1,1,1,0]
]

# Now we'll compute y for each x and compare
for i in range(len(x_data)):
    computed_y = compute_y(x_data[i])
    if computed_y == y_data[i]:
        print('Success')
    else:
        print('Failure')