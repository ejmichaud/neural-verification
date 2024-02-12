# Define function to compute y from x
def compute_y(x):
    y = []
    for i in range(len(x)):
        if x[i] > 0:
            y.append(x[i])
        else:
            # Look ahead for the next positive number in x
            j = i
            while j < len(x) and x[j] <= 0:
                j += 1
            if j < len(x):
                y.append(abs(x[i]) + x[j])
            else:
                # If there are no more positive numbers, just take the absolute value
                y.append(abs(x[i]))
    return y

# List of x and y pairs
x_lists = [
    [42,-33,-24,-86,-74,35,-80,24,50,13],
    [78,-86,-90,54,-69,72,15,-5,67,6],
    [49,-24,-27,11,-1,-87,41,69,-13,-81],
    [72,-20,-25,29,33,64,39,76,-68,-90],
    [86,-78,77,19,7,23,43,-6,-7,-23]
]

y_lists = [
    [42,75,9,62,12,109,115,104,26,37],
    [78,164,4,144,123,141,57,20,72,61],
    [49,73,3,38,12,86,128,28,82,68],
    [72,92,5,54,4,31,25,37,144,22],
    [86,164,155,58,12,16,20,49,1,16]
]

# Iterate through each pair of x and y lists
for x, y_expected in zip(x_lists, y_lists):
    y_computed = compute_y(x)
    if y_computed == y_expected:
        print('Success')
    else:
        print('Failure')
        print('Computed y:', y_computed)
        print('Expected y:', y_expected)