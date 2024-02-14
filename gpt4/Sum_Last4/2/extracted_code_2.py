def compute_y(x):
    y = [x[0]]  # Start with the first element
    for i in range(1, len(x)):
        y.append(y[-1] + x[i])  # Append the cumulative sum
    return y

# List of x arrays from the table
list_of_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Corresponding list of y arrays from the table
list_of_y = [
    [42,109,185,199,183,151,95,105,129,107],
    [78,92,102,156,109,167,172,213,249,183],
    [49,125,198,209,259,196,164,222,210,216],
    [72,152,227,256,217,201,165,212,211,157],
    [86,108,185,204,125,126,92,167,253,307]
]

# Check the computed y against the provided y
for i, x in enumerate(list_of_x):
    computed_y = compute_y(x)
    if computed_y == list_of_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure: expected {list_of_y[i]} but got {computed_y}')