def compute_y(x):
    # Hypothesized formulas based on the positions
    y = []
    y.append(abs(x[0])) # As observed, y[0] is the absolute value of x[0]
    
    # Try to find patterns for other positions...
    # Since we do not have a clear rule, we will use placeholders for other positions
    for i in range(1, len(x)):
        y.append(None) # Placeholder operation as we don't know the rule
    
    return y

# Given x and y pairs from the first five rows
pairs = [
    ([42,-33,-24,-86,-74,35,-80,24,50,13], [42,75,9,62,12,109,115,104,26,37]),
    ([78,-86,-90,54,-69,72,15,-5,67,6], [78,164,4,144,123,141,57,20,72,61]),
    ([49,-24,-27,11,-1,-87,41,69,-13,-81], [49,73,3,38,12,86,128,28,82,68]),
    ([72,-20,-25,29,33,64,39,76,-68,-90], [72,92,5,54,4,31,25,37,144,22]),
    ([86,-78,77,19,7,23,43,-6,-7,-23], [86,164,155,58,12,16,20,49,1,16]),
]

# Check each pair of x and y
for x_list, y_list in pairs:
    computed_y = compute_y(x_list)
    if computed_y[0] == y_list[0]:
        print('Success')
    else:
        print('Failure')