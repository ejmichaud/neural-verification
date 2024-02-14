def compute_y(x):
    length = len(x)
    y = []
    for i in range(length):
        left = x[i - 1] if i - 1 >= 0 else x[length - 1]
        right = x[i + 1] if i + 1 < length else x[0]
        y.append(left ^ right)
    return y

# List of x and y pairs for the first five rows
pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,1,1,0,1,1,1,0,1]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,1,1,0,0,1,0]),
    ([1,0,1,1,1,1,1,1,1,1], [1,1,0,0,1,1,1,1,1,1]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,1,0,1,0,0,1,1,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,1,0,1,1,1,0,0,0])
]

for x, expected_y in pairs:
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')