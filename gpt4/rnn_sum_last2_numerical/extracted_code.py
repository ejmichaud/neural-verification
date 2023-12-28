# Input data for the first 5 rows
input_data = [
    ([42, 67, 76, 14, 26, 35, 20, 24, 50, 13], [42, 109, 143, 90, 40, 61, 55, 44, 74, 63]),
    ([78, 14, 10, 54, 31, 72, 15, 95, 67, 6], [78, 92, 24, 64, 85, 103, 87, 110, 162, 73]),
    ([49, 76, 73, 11, 99, 13, 41, 69, 87, 19], [49, 125, 149, 84, 110, 112, 54, 110, 156, 106]),
    ([72, 80, 75, 29, 33, 64, 39, 76, 32, 10], [72, 152, 155, 104, 62, 97, 103, 115, 108, 42]),
    ([86, 22, 77, 19, 7, 23, 43, 94, 93, 77], [86, 108, 99, 96, 26, 30, 66, 137, 187, 170]),
]

def compute_y_from_x(list_x):
    """
    Function that takes list x and computes list y based on the cumulative sum.
    """
    list_y = []
    cumulative_sum = 0
    for x in list_x:
        cumulative_sum += x
        list_y.append(cumulative_sum)
    return list_y

# Initialize the result variable as 'Success'
result = 'Success'

# Process the input data and check the output
for row in input_data:
    x, expected_y = row
    computed_y = compute_y_from_x(x)
    
    # Check if lists match
    if computed_y != expected_y:
        result = 'Failure'
        break  # No need to check further if one has already failed

# Print result
print(result)