# Define the lists x and y for comparison
input_output_pairs = [
    ([5, -8, -6, 9, 8, 7, 9, -5, -3, 4], [5, 2, -7, -7, 1, 16, 40, 59, 75, 95]),
    ([-3, 9, 10, 8, 5, 6, 10, 3, -1, 0], [-3, 3, 19, 43, 72, 107, 152, 200, 247, 294]),
    ([-9, 4, -3, 7, 7, 9, -2, 5, 6, 6], [-9, -14, -22, -23, -17, -2, 11, 29, 53, 83]),
    ([-1, 2, -3, -3, -5, -3, 1, 7, 3, 5], [-1, 0, -2, -7, -17, -30, -42, -47, -49, -46]),
    ( [4, 5, 9, 2, -4, 1, 4, 6, 2, -5], [4, 13, 31, 51, 67, 84, 105, 132, 161, 185])
]

def compute_y_from_x(x_list):
    y_list = []
    # Start with the first element of x for y
    for i, x in enumerate(x_list):
        if i == 0:
            y_list.append(x)
        else:
            # Each element in y is a sum of the previous y and the current x
            y_list.append(y_list[-1] + x)
    return y_list

# Process each list x and check against list y
success = True
for x, expected_y in input_output_pairs[:5]:
    computed_y = compute_y_from_x(x)
    if computed_y != expected_y:
        success = False
        print("Failure")
        break

if success:
    print("Success")