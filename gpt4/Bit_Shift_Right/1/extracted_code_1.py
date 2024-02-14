# Define the list of lists x and y
list_x = [
    [0,1,0,0,0,1,0,0,0,1],
    [0,0,0,0,1,0,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,0,1,0,0,0],
    [0,0,1,1,1,1,1,0,1,1]
]

list_y = [
    [0,0,1,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,1,1,1],
    [0,1,0,1,1,1,1,1,1,1],
    [0,0,0,1,1,1,0,1,0,0],
    [0,0,0,1,1,1,1,1,0,1]
]

# Define the function to compute y from x
def shift_elements(x):
    y = x[1:] + [x[0]]
    return y

# Apply the function and check if the output matches list y
success = True
for i in range(5):
    computed_y = shift_elements(list_x[i])
    if computed_y == list_y[i]:
        print(f'Row {i+1}: Success')
    else:
        print(f'Row {i+1}: Failure')
        success = False

# Print overall result
if success:
    print('All tests passed!')
else:
    print('There were some errors.')