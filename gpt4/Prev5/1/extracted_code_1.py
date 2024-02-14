# Define the function to compute y from x
def compute_y(x):
    # Create a list of 5 zeros
    y = [0] * 5
    # Extend the list y by the contents of x, starting from the 6th element (index 5)
    y.extend(x[:-5])
    return y

# List of x and y pairs for the first 5 rows
x_values = [
    [42,67,76,14,26,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19,72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77,70,9,70,39,86,99,15,84,78,8],
    [66,30,40,60,70,61,23,20,11,61,77,89,84,53,48,9,83,7,58,91],
    [14,91,36,3,82,90,89,28,55,33,27,47,65,89,41,45,61,39,61,64]
]

y_values = [
    [0,0,0,0,0,42,67,76,14,26,35,20,24,50,13,78,14,10,54,31],
    [0,0,0,0,0,49,76,73,11,99,13,41,69,87,19,72,80,75,29,33],
    [0,0,0,0,0,86,22,77,19,7,23,43,94,93,77,70,9,70,39,86],
    [0,0,0,0,0,66,30,40,60,70,61,23,20,11,61,77,89,84,53,48],
    [0,0,0,0,0,14,91,36,3,82,90,89,28,55,33,27,47,65,89,41]
]

# Process each row and check if the computed y matches the given y
for i in range(len(x_values)):
    computed_y = compute_y(x_values[i])
    if computed_y == y_values[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')