def compute_y(x):
    # Basic idea is to check if each 1 in x becomes 0 in y and each 0 becomes 1 in y in the first 5 rows
    return [1 - xi for xi in x]

def check_transformations(lists_x, lists_y):
    for i in range(5):
        computed_y = compute_y(lists_x[i])
        if computed_y == lists_y[i]:
            print(f"Row {i+1} Success")
        else:
            print(f"Row {i+1} Failure")
            print(f"Expected: {lists_y[i]}")
            print(f"Computed: {computed_y}")

lists_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

lists_y = [
    [0,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,0,1,1,1],
    [1,1,0,1,1,0,0,0,0,0],
    [0,0,1,0,1,1,1,0,1,1],
    [0,0,1,0,1,0,0,1,1,1]
]

# Let's check if our basic attempt works
check_transformations(lists_x, lists_y)