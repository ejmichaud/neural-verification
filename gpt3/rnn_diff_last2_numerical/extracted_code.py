def compute_list_y(list_x):
    list_y = [2*list_x[i] - list_x[i+1] for i in range(len(list_x) - 1)]
    return list_y

# The given lists x and y
list_x = [
    [42,-33,-24,-86,-74,35,-80,24,50,13],
    [78,-86,-90,54,-69,72,15,-5,67,6],
    [49,-24,-27,11,-1,-87,41,69,-13,-81],
    [72,-20,-25,29,33,64,39,76,-68,-90],
    [86,-78,77,19,7,23,43,-6,-7,-23]
]
list_y_expected = [
    [42,-75,9,-62,12,109,-115,104,26,-37],
    [78,-164,-4,144,-123,141,-57,-20,72,-61],
    [49,-73,-3,38,-12,-86,128,28,-82,-68],
    [72,-92,-5,54,4,31,-25,37,-144,-22],
    [86,-164,155,-58,-12,16,20,-49,-1,-16]
]

# Check if the output matches list y
for i in range(5):
    list_y_computed = compute_list_y(list_x[i])
    if list_y_computed == list_y_expected[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")