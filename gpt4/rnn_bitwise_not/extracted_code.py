# Define the lists x and y as given
x_lists = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

y_lists = [
    [1,0,1,1,1,0,1,1,1,0],
    [1,1,1,1,0,1,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,1,0,1,1,1],
    [1,1,0,0,0,0,0,1,0,0]
]

# Naively invert each bit to see if it matches the y list
def naive_invert(x):
    return [1 - xi for xi in x]

# Check for each list if the naive inversion matches the given y list
for i in range(len(x_lists)):
    computed_y = naive_invert(x_lists[i])
    if computed_y == y_lists[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')