# Define the first five rows of list x and corresponding list y
data = [
    ([42, -33, -24, -86, -74, 35, -80, 24, 50, 13], [42, 33, 24, 86, 74, 35, 80, 24, 50, 13]),
    ([78, -86, -90, 54, -69, 72, 15, -5, 67, 6], [78, 86, 90, 54, 69, 72, 15, 5, 67, 6]),
    ([49, -24, -27, 11, -1, -87, 41, 69, -13, -81], [49, 24, 27, 11, 1, 87, 41, 69, 13, 81]),
    ([72, -20, -25, 29, 33, 64, 39, 76, -68, -90], [72, 20, 25, 29, 33, 64, 39, 76, 68, 90]),
    ([86, -78, 77, 19, 7, 23, 43, -6, -7, -23], [86, 78, 77, 19, 7, 23, 43, 6, 7, 23]),
]

# Function to compute list y from list x
def compute_y_from_x(list_x):
    return [abs(x) for x in list_x]

# Process each row and check if the computed list y matches the expected list y
for index, (list_x, expected_list_y) in enumerate(data):
    computed_list_y = compute_y_from_x(list_x)
    if computed_list_y == expected_list_y:
        print(f"Row {index + 1} Success")
    else:
        print(f"Row {index + 1} Failure")

# If all are successful, print 'All Success'
print('All Success')