# Define the first 5 rows of list x
list_x = [
    [42,-33,-24,-86,-74,35,-80,24,50,13],
    [78,-86,-90,54,-69,72,15,-5,67,6],
    [49,-24,-27,11,-1,-87,41,69,-13,-81],
    [72,-20,-25,29,33,64,39,76,-68,-90],
    [86,-78,77,19,7,23,43,-6,-7,-23]
]

# Define the expected first 5 rows of list y based on the problem description
expected_list_y = [
    [42,33,24,86,74,35,80,24,50,13],
    [78,86,90,54,69,72,15,5,67,6],
    [49,24,27,11,1,87,41,69,13,81],
    [72,20,25,29,33,64,39,76,68,90],
    [86,78,77,19,7,23,43,6,7,23]
]

# Compute list y using the absolute value function and compare with the expected list y
for i, x in enumerate(list_x):
    computed_y = [abs(num) for num in x]
    # Check if the computed list y matches the expected list y
    if computed_y == expected_list_y[i]:
        print("Row", i + 1, "Success")
    else:
        print("Row", i + 1, "Failure")