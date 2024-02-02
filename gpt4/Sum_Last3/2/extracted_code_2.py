# Given list x and y pairs for the first five rows
pairs = [
    ([42,67,76,14,26,35,20,24,50,13], [42,109,185,157,116,75,81,79,94,87]),
    ([78,14,10,54,31,72,15,95,67,6], [78,92,102,78,95,157,118,182,177,168]),
    ([49,76,73,11,99,13,41,69,87,19], [49,125,198,160,183,123,153,123,197,175]),
    ([72,80,75,29,33,64,39,76,32,10], [72,152,227,184,137,126,136,179,147,118]),
    ([86,22,77,19,7,23,43,94,93,77], [86,108,185,118,103,49,73,160,230,264]),
]

# Function to compute list y from list x
def compute_y(x):
    y = []
    cumulative_sum = 0
    for value in x:
        cumulative_sum += value
        y.append(cumulative_sum)
    return y

# Process each pair and check the result
for x_values, expected_y_values in pairs:
    computed_y_values = compute_y(x_values)
    if computed_y_values == expected_y_values:
        print('Success')
    else:
        print('Failure')