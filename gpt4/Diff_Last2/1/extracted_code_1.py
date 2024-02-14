# The sample data from the question.
data = [
    ([42, -33, -24, -86, -74, 35, -80, 24, 50, 13], [42, -75, 9, -62, 12, 109, -115, 104, 26, -37]),
    ([78, -86, -90, 54, -69, 72, 15, -5, 67, 6], [78, -164, -4, 144, -123, 141, -57, -20, 72, -61]),
    ([49, -24, -27, 11, -1, -87, 41, 69, -13, -81], [49, -73, -3, 38, -12, -86, 128, 28, -82, -68]),
    ([72, -20, -25, 29, 33, 64, 39, 76, -68, -90], [72, -92, -5, 54, 4, 31, -25, 37, -144, -22]),
    ([86, -78, 77, 19, 7, 23, 43, -6, -7, -23], [86, -164, 155, -58, -12, 16, 20, -49, -1, -16]),
]

def compute_y(x):
    return [x[i] + x[0] - i for i in range(len(x))]

def check_success(computed_y, expected_y):
    return 'Success' if computed_y == expected_y else 'Failure'

for x_list, y_list in data:
    computed_y = compute_y(x_list)
    result = check_success(computed_y, y_list)
    print(result)