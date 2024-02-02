# Define the lists of x and y
list_xs = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

list_ys = [
    [42, 109, 185, 157, 116, 75, 81, 79, 94, 87],
    [78, 92, 102, 78, 95, 157, 118, 182, 177, 168],
    [49, 125, 198, 160, 183, 123, 153, 123, 197, 175],
    [72, 152, 227, 184, 137, 126, 136, 179, 147, 118],
    [86, 108, 185, 118, 103, 49, 73, 160, 230, 264]
]

def compute_cumulative_sum(x):
    return [sum(x[:i+1]) for i in range(len(x))]

# Check if the cumulative sum matches the y lists
for index, x in enumerate(list_xs):
    computed_y = compute_cumulative_sum(x)
    if computed_y == list_ys[index]:
        print(f"Row {index+1} Success")
    else:
        print(f"Row {index+1} Failure")

# Let's check to see if there's any matching using the hypothesis
all_success = True
for i in range(5):
    computed_y = compute_cumulative_sum(list_xs[i])
    if computed_y != list_ys[i]:
        all_success = False
        break

if all_success:
    print('Success')
else:
    print('Failure')