def f(x_i, x_prev, x_next):
    if x_i == 0:
        if x_next == 1 and (x_prev is None or x_prev != 2):
            return 0
        else:
            return 0
    elif x_i == 1:
        if (x_prev == 2 and x_next == 2) or (x_prev is not None and x_prev == 0 and x_next == 0):
            return 0
        else:
            return 1
    elif x_i == 2:
        if x_prev != 2:
            return 2
        else:
            return 0

def compute_y(x_list):
    y_computed = []
    for i in range(len(x_list)):
        x_prev = x_list[i-1] if i-1 >= 0 else None
        x_next = x_list[i+1] if i+1 < len(x_list) else None
        y_computed.append(f(x_list[i], x_prev, x_next))
    return y_computed

# Here are the first five rows of x and y
x_rows = [
    [0,2,1,1,0,2,1,2,1,2],
    [1,1,2,0,0,1,2,1,0,1],
    [1,2,1,2,2,1,2,0,1,1],
    [0,0,1,1,2,1,2,2,1,0],
    [2,0,1,0,0,2,2,1,0,2]
]

y_rows = [
    [0,0,0,1,0,0,1,1,1,1],
    [1,1,1,1,0,1,1,1,1,1],
    [1,1,1,1,2,1,2,2,1,1],
    [0,0,0,0,0,1,1,1,1,1],
    [2,0,0,0,0,0,0,0,0,0]
]

# Compute and check y for the first five rows
for i in range(5):
    x = x_rows[i]
    expected_y = y_rows[i]
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}: computed {computed_y} but expected {expected_y}")