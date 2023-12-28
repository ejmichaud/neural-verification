# Define the compute_y function
def compute_y(x):
    # Initialize y with two zeros and extend with x except the last two elements
    y = [0, 0] + x[:-2]
    return y

# Given data for the first 5 rows
list_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

list_y_expected = [
    [0,0,42,67,76,14,26,35,20,24],
    [0,0,78,14,10,54,31,72,15,95],
    [0,0,49,76,73,11,99,13,41,69],
    [0,0,72,80,75,29,33,64,39,76],
    [0,0,86,22,77,19,7,23,43,94]
]

# Loop through each row and compute y, then compare the result with the expected y
for i in range(len(list_x)):
    y_computed = compute_y(list_x[i])
    if y_computed == list_y_expected[i]:
        print(f'Row {i + 1} Success')
    else:
        print(f'Row {i + 1} Failure')