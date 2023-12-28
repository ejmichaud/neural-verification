# Define the input list of lists x
list_of_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

# Define the input list of lists y
list_of_ys = [
    [0, 0, 0, 0, 42, 67, 76, 14, 26, 35],
    [0, 0, 0, 0, 78, 14, 10, 54, 31, 72],
    [0, 0, 0, 0, 49, 76, 73, 11, 99, 13],
    [0, 0, 0, 0, 72, 80, 75, 29, 33, 64],
    [0, 0, 0, 0, 86, 22, 77, 19, 7, 23]
]

# Process each x list to compute y and compare with the provided y
for i in range(len(list_of_xs)):
    x = list_of_xs[i]
    computed_y = ([0, 0, 0, 0] + x[:-4]) # Compute y by adding four zeroes and then the elements of x shifted by 4
    provided_y = list_of_ys[i]

    # Check if the computed y matches the provided y
    if computed_y == provided_y:
        print('Success')
    else:
        print('Failure')

# If every comparison was 'Success', it means all computed y lists match their corresponding provided y lists