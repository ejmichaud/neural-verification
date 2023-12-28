# Define the input lists
x_lists = [
    [42,-33,-24,-86,-74,35,-80,24,50,13],
    [78,-86,-90,54,-69,72,15,-5,67,6],
    [49,-24,-27,11,-1,-87,41,69,-13,-81],
    [72,-20,-25,29,33,64,39,76,-68,-90],
    [86,-78,77,19,7,23,43,-6,-7,-23]
]
y_lists = [
    [42,75,9,62,12,109,115,104,26,37],
    [78,164,4,144,123,141,57,20,72,61],
    [49,73,3,38,12,86,128,28,82,68],
    [72,92,5,54,4,31,25,37,144,22],
    [86,164,155,58,12,16,20,49,1,16]
]

# Compute list y and check if it matches the given list y
for i in range(5):
    x = x_lists[i]
    computed_y = [abs(x[j]) + abs(x[j+1]) for j in range(len(x)-1)]
    result = "Success" if computed_y == y_lists[i] else "Failure"
    print(f"Computed List y: {computed_y}, Expected List y: {y_lists[i]}, Result: {result}")