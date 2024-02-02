# Define the input lists of x
input_lists_x = [
    [0,7,7,4,7,7,2,2,6,0],
    [0,6,7,5,4,4,2,6,3,0],
    [6,7,1,5,3,3,1,6,3,3],
    [3,6,0,6,5,6,4,7,6,6],
    [6,0,3,5,3,3,1,3,1,3]
]

# Define the corresponding output lists of y
output_lists_y = [
    [1,0,1,0,0,1,0,1,0,0],
    [1,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0,0,0,0]
]

# Function to compute list y from list x based on observed pattern
def compute_list_y(list_x):
    # A '1' in y occurs where there is a '7' in x
    return [1 if element == 7 else 0 for element in list_x]

# Check whether the computed list y matches the given list y
success = True
for i in range(5):
    computed_list_y = compute_list_y(input_lists_x[i])
    if computed_list_y == output_lists_y[i]:
        print(f'Row {i+1} - Success')
    else:
        print(f'Row {i+1} - Failure')
        success = False

# Print the final result
if success:
    print('All tests passed: Success')
else:
    print('Some tests failed: Failure')