# List of x and y pairs
list_pairs = [
    ([6,3,4,6,2,7,4,4,6,1], [6,1,5,3,5,4,0,4,2,3]),
    ([2,6,2,2,7,4,3,7,7,2], [2,0,2,4,3,7,2,1,0,2]),
    ([5,4,1,7,3,5,5,1,7,3], [5,1,2,1,4,1,6,7,6,1]),
    ([4,0,3,1,5,4,3,0,0,2], [4,4,7,0,5,1,4,4,4,6]),
    ([2,6,1,7,3,3,7,6,5,5], [2,0,1,0,3,6,5,3,0,5]),
]

# Function to compute y from x using the formula y = x % 8
def compute_y_from_x(x_list):
    return [x % 8 for x in x_list]

# Main program flow
success = True

for x_values, expected_y_values in list_pairs:
    computed_y_values = compute_y_from_x(x_values)
    if computed_y_values == expected_y_values:
        print('Success')
    else:
        print('Failure')
        success = False

if success:
    print("The formula y = x % 8 works for all 5 rows!")
else:
    print("The formula didn't match all the rows.")