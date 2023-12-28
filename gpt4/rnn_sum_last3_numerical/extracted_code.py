def compute_y_from_x(x):
    y = []
    cumulative_sum = 0
    for element in x:
        cumulative_sum += element
        y.append(cumulative_sum)
    return y

# Data from the table for the first 5 rows
input_output_pairs = [
    ([42,67,76,14,26,35,20,24,50,13], [42,109,185,157,116,75,81,79,94,87]),
    ([78,14,10,54,31,72,15,95,67,6], [78,92,102,78,95,157,118,182,177,168]),
    ([49,76,73,11,99,13,41,69,87,19], [49,125,198,160,183,123,153,123,197,175]),
    ([72,80,75,29,33,64,39,76,32,10], [72,152,227,184,137,126,136,179,147,118]),
    ([86,22,77,19,7,23,43,94,93,77], [86,108,185,118,103,49,73,160,230,264]),
]

# Iterate through each x, y pair and compute y from x to check if it matches the provided y
for x, expected_y in input_output_pairs:
    computed_y = compute_y_from_x(x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')