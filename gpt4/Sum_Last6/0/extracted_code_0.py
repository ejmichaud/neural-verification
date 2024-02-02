# Define the list of x values for the first five rows
x_values = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70, 39, 86, 99, 15, 84, 78, 8],
    [66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84, 53, 48, 9, 83, 7, 58, 91],
    [14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65, 89, 41, 45, 61, 39, 61, 64]
]

# Define the list of y values for the first five rows (for comparison)
y_values_expected = [
    [42, 109, 185, 199, 225, 260, 238, 195, 169, 168, 220, 199, 189, 219, 200, 259, 196, 277, 334, 286],
    [49, 125, 198, 209, 308, 321, 313, 306, 320, 328, 301, 368, 402, 362, 308, 353, 320, 316, 273, 254],
    [86, 108, 185, 204, 211, 234, 191, 263, 279, 337, 400, 386, 413, 358, 351, 373, 318, 393, 401, 370],
    [66, 96, 136, 196, 266, 327, 284, 274, 245, 246, 253, 281, 342, 375, 412, 360, 366, 284, 258, 296],
    [14, 105, 141, 144, 226, 316, 391, 328, 347, 377, 322, 279, 255, 316, 302, 314, 348, 340, 336, 311]
]

# A function to compute list y from list x
def compute_y(x):
    y = []
    running_total = 0
    for value in x:
        running_total += value
        y.append(running_total)
    return y

# Iterate over the first five rows and run the check
for i in range(5):
    y_computed = compute_y(x_values[i])
    if y_computed == y_values_expected[i]:
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")

# If all rows are successful, print 'Success', otherwise print 'Failure'
results = all(compute_y(x_values[i]) == y_values_expected[i] for i in range(5))
if results:
    print('Success')
else:
    print('Failure')