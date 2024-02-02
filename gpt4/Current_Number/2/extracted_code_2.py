# Define the identity function
def f(x):
    return x

# Define the lists of x and y
lists_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

lists_y = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Process the first 5 rows and check if the computed y matches the original y
for i in range(5):
    computed_y = f(lists_x[i])
    if computed_y == lists_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")