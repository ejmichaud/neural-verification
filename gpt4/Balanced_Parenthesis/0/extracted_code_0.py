def compute_y(x):
    y = []
    for i in range(len(x)):
        # Check if the current element is 1 and it is not part of a sequence of 1s
        if x[i] == 1 and (i == 0 or x[i-1] == 0) and (i == len(x) - 1 or x[i+1] == 0):
            y.append(1)
        else:
            y.append(0)
    return y

# Given lists of x and expected outputs y
input_output_pairs = [
    ([0,1,0,0,0,1,0,0,0,1], [0,1,0,0,0,0,0,0,0,0]),
    ([0,0,0,0,1,0,1,1,1,0], [0,0,0,0,0,0,0,0,0,0]),
    ([1,0,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0], [0,0,0,1,0,0,0,0,0,0]),
    ([0,0,1,1,1,1,1,0,1,1], [0,0,0,1,0,0,0,0,0,0]),
]

# Check each pair and print Success/Failure
for x_list, expected_y_list in input_output_pairs:
    computed_y_list = compute_y(x_list)
    if computed_y_list == expected_y_list:
        print("Success")
    else:
        print("Failure")