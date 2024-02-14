# Define lists x and y (five rows each, as per the example given)
lists_x = [
    [5, -8], [-6, 9], [8, 7], [9, -5], [-3, 4], 
    [-3, 9], [10, 8], [5, 6], [10, 3], [-1, 0]
]
lists_y = [
    [5, -8], [12, -2], [21, 18], [19, 42], [-10, 68], 
    [-68, 74], [-122, 30], [-127, -62], [-30, -156], [160, -153]
]

# Define a function to compute y from x using a guessed formula
def compute_y(x):
    # This is a placeholder transformation based on observation of some elements
    # Replace this with the correct transformation function once we find the pattern
    return [x[0]*x[1], x[0] + x[1]]

# Iterate through the five rows and check if the transformation matches
success = True
for i in range(5):
    if lists_y[i] != compute_y(lists_x[i]):
        success = False

print('Success' if success else 'Failure')