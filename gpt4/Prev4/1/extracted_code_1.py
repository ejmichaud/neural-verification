# x lists provided
x_lists = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# y lists provided 
y_lists = [
    [0,0,0,0,42,67,76,14,26,35],
    [0,0,0,0,78,14,10,54,31,72],
    [0,0,0,0,49,76,73,11,99,13],
    [0,0,0,0,72,80,75,29,33,64],
    [0,0,0,0,86,22,77,19,7,23]
]

def compute_y(x):
    # Initialize y with five 0's and then extend using the elements of x excluding the last five
    y = [0] * 5 + x[:-5]
    return y

# Process each row
for row in range(5):  # Only process the first five rows
    computed_y = compute_y(x_lists[row])
    if computed_y == y_lists[row]:
        print(f'Row {row + 1} Success')
    else:
        print(f'Row {row + 1} Failure')

        # Optionally print the output to verify if there's a mismatch
        #print("Expected y:", y_lists[row])
        #print("Computed y:", computed_y)