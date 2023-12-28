import numpy as np

# Define the lists of x and y values
list_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

computed_y = []
actual_y = [
    [42, 109, 185, 199, 225, 260, 280, 304, 354, 367],
    [78, 92, 102, 156, 187, 259, 274, 369, 436, 442],
    [49, 125, 198, 209, 308, 321, 362, 431, 518, 537],
    [72, 152, 227, 256, 289, 353, 392, 468, 500, 510],
    [86, 108, 185, 204, 211, 234, 277, 371, 464, 541]
]

# Compute list y using a simple linear regression
for i in range(5):
    x = np.array(list_x[i])
    y = 2 * x  # Example computation (could be replaced with regression formula)
    computed_y.append(y.tolist())

# Check the computed y values against the actual y values
success = all(np.array_equal(computed_y[i], actual_y[i]) for i in range(5))

if success:
    print("Success")
else:
    print("Failure")

print("Computed Y:", computed_y)
print("Actual Y:", actual_y)