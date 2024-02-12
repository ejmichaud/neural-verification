def compute_y(x_list):
    y = [x_list[0]]
    for i in range(1, len(x_list)):
        y.append(y[-1] + x_list[i])
    return y
    
# Define the first 5 rows of x and y as given in the table
x_data = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

y_data = [
    [42, 109, 185, 199, 223, 258, 278, 302, 352, 365],
    [78, 92, 102, 156, 187, 259, 274, 369, 436, 442],
    [49, 125, 198, 209, 308, 321, 362, 431, 518, 537],
    [72, 152, 227, 256, 289, 353, 392, 468, 500, 510],
    [86, 108, 185, 204, 211, 234, 277, 371, 464, 541]
]

# Run the computation and check the results for the first 5 rows
for i in range(len(x_data)):
    computed_y = compute_y(x_data[i])
    # Compare the computed y with the given y
    if computed_y == y_data[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")