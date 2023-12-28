# Define the given lists of x and y for the first 5 rows
list_of_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70, 39, 86, 99, 15, 84, 78, 8],
    [66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84, 53, 48, 9, 83, 7, 58, 91],
    [14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65, 89, 41, 45, 61, 39, 61, 64]
]

list_of_y = [
    [42, 109, 185, 199, 225, 260, 280, 262, 245, 182, 246, 234, 209, 243, 250, 272, 274, 291, 344, 340],
    [49, 125, 198, 209, 308, 321, 362, 382, 393, 339, 400, 381, 443, 431, 395, 372, 392, 396, 348, 283],
    [86, 108, 185, 204, 211, 234, 277, 285, 356, 356, 407, 409, 456, 452, 444, 450, 388, 402, 471, 409],
    [66, 96, 136, 196, 266, 327, 350, 304, 285, 306, 323, 342, 365, 395, 423, 421, 443, 373, 342, 349],
    [14, 105, 141, 144, 226, 316, 405, 419, 383, 380, 404, 369, 344, 344, 357, 347, 375, 387, 401, 400]
]

# Function to compute cumulative sum of list x
def compute_cumulative_sum(x_list):
    return [sum(x_list[:i+1]) for i in range(len(x_list))]

# Loop through the first 5 rows and perform the check
for i in range(5):
    computed_y = compute_cumulative_sum(list_of_x[i])
    if computed_y == list_of_y[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')

# Run the program
if __name__ == "__main__":
    for i in range(5):
        computed_y = compute_cumulative_sum(list_of_x[i])
        if computed_y == list_of_y[i]:
            print(f'Row {i+1} Success')
        else:
            print(f'Row {i+1} Failure')