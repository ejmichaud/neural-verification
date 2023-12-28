# Define the function to compute list y from list x
def compute_y(x):
    y = []
    running_sum = 0
    for i in range(len(x)):
        running_sum += x[i]
        y.append(x[i] + 2 * running_sum)
    return y

# Input data
list_x = [
    [42,67,76,14,26,35,20,24,50,13],
    [78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19],
    [72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77]
]

# Expected output
list_y_expected = [
    [42,109,185,157,116,75,81,79,94,87],
    [78,92,102,78,95,157,118,182,177,168],
    [49,125,198,160,183,123,153,123,197,175],
    [72,152,227,184,137,126,136,179,147,118],
    [86,108,185,118,103,49,73,160,230,264]
]

# Compute list y and compare with expected output
for i in range(5):
    y_computed = compute_y(list_x[i])
    if y_computed == list_y_expected[i]:
        print("Success")
    else:
        print("Failure")