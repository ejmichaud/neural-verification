# Given lists of x and y
lists_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

lists_y = [
    [42, 109, 185, 199, 225, 260, 280, 304, 354, 367],
    [78, 92, 102, 156, 187, 259, 274, 369, 436, 442],
    [49, 125, 198, 209, 308, 321, 362, 431, 518, 537],
    [72, 152, 227, 256, 289, 353, 392, 468, 500, 510],
    [86, 108, 185, 204, 211, 234, 277, 371, 464, 541]
]

def compute_cumulative_sum(lst_x):
    cumulative_sum = 0
    lst_y = []
    for x in lst_x:
        cumulative_sum += x
        lst_y.append(cumulative_sum)
    return lst_y

# Iterate through the first five rows
success = True
for i in range(5):
    computed_y = compute_cumulative_sum(lists_x[i])
    # Check if the computed y list matches the given y list
    if computed_y == lists_y[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")
        success = False

# Check final result
if success:
    print("All rows matched. Overall Result: Success")
else:
    print("One or more rows did not match. Overall Result: Failure")