def cumulative_sum(x):
    y = []
    current_sum = 0
    for number in x:
        current_sum += number
        y.append(current_sum)
    return y

# Given data for the first 5 rows
data = [
    ([42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6], 
     [42, 109, 185, 199, 225, 260, 238, 195, 169, 168, 220, 199, 189, 219, 200, 259, 196, 277, 334, 286]),
    ([49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
     [49, 125, 198, 209, 308, 321, 313, 306, 320, 328, 301, 368, 402, 362, 308, 353, 320, 316, 273, 254]),
    ([86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70, 39, 86, 99, 15, 84, 78, 8],
     [86, 108, 185, 204, 211, 234, 191, 263, 279, 337, 400, 386, 413, 358, 351, 373, 318, 393, 401, 370]),
    ([66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84, 53, 48, 9, 83, 7, 58, 91],
     [66, 96, 136, 196, 266, 327, 284, 274, 245, 246, 253, 281, 342, 375, 412, 360, 366, 284, 258, 296]),
    ([14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65, 89, 41, 45, 61, 39, 61, 64],
     [14, 105, 141, 144, 226, 316, 391, 328, 347, 377, 322, 279, 255, 316, 302, 314, 348, 340, 336, 311])
]

# Check each row and print 'Success' if the computed y matches the given y
def check_cumulative_sum(data):
    for row_count, (x, given_y) in enumerate(data):
        computed_y = cumulative_sum(x)
        if computed_y == given_y:
            print(f'Row {row_count + 1} Success')
        else:
            print(f'Row {row_count + 1} Failure')

# Call the function to check the cumulative sum for the first 5 rows
check_cumulative_sum(data)