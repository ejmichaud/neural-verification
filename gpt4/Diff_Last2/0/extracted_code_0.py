def compute_y_from_x(x_list):
    y_computed = [x_list[0]]
    for n in range(1, len(x_list)):
        y_computed.append(x_list[0] + x_list[n])
    return y_computed

# Dataset for the first 5 rows
dataset = [
    ([42,-33,-24,-86,-74,35,-80,24,50,13], [42,-75,9,-62,12,109,-115,104,26,-37]),
    ([78,-86,-90,54,-69,72,15,-5,67,6], [78,-164,-4,144,-123,141,-57,-20,72,-61]),
    ([49,-24,-27,11,-1,-87,41,69,-13,-81], [49,-73,-3,38,-12,-86,128,28,-82,-68]),
    ([72,-20,-25,29,33,64,39,76,-68,-90], [72,-92,-5,54,4,31,-25,37,-144,-22]),
    ([86,-78,77,19,7,23,43,-6,-7,-23], [86,-164,155,-58,-12,16,20,-49,-1,-16]),
]

# Processing each row and checking the results
for row in dataset:
    x_list, expected_y_list = row
    computed_y_list = compute_y_from_x(x_list)
    # Check if the computed y list matches the expected y list
    if computed_y_list == expected_y_list:
        print("Success")
    else:
        print("Failure")