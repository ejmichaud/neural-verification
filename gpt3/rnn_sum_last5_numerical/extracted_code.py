import numpy as np

# Given lists
list_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70, 39, 86, 99, 15, 84, 78, 8],
    [66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84, 53, 48, 9, 83, 7, 58, 91],
    [14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65, 89, 41, 45, 61, 39, 61, 64]
]

list_y = [
    [42, 109, 185, 199, 225, 218, 171, 119, 155, 142, 185, 179, 165, 169, 187, 181, 182, 267, 280, 255],
    [49, 125, 198, 209, 308, 272, 237, 233, 309, 229, 288, 327, 333, 275, 289, 281, 240, 241, 244, 221],
    [86, 108, 185, 204, 211, 148, 169, 186, 260, 330, 377, 343, 319, 265, 274, 303, 309, 323, 362, 284],
    [66, 96, 136, 196, 266, 261, 254, 234, 185, 176, 192, 258, 322, 364, 351, 283, 277, 200, 205, 248],
    [14, 105, 141, 144, 226, 302, 300, 292, 344, 295, 232, 190, 227, 261, 269, 287, 301, 275, 247, 270]
]

# Function to compute y from x using polynomial regression
def compute_y_from_x(x):
    coef = np.polyfit(range(len(x)), x, len(x) - 1)
    y = [np.polyval(coef, i) for i in range(len(x))]
    return y

# Compute y for each row in list_x and check if it matches list_y
for i in range(5):
    computed_y = compute_y_from_x(list_x[i])
    if np.array_equal(computed_y, list_y[i]):
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")