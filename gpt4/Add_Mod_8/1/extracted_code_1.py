def compute_y(x_list):
    y_computed = []
    for i, x in enumerate(x_list):
        # Applying our hypothesized formula here
        if x == 2 and len(y_computed) > 0:
            y_computed.append(y_computed[-1] - 2)
        elif x == x_list[i-1] and i > 0:
            y_computed.append(y_computed[-1] + 2)
        else:
            y_computed.append(x)
    return y_computed

# Given list of x's and y's
x_lists = [
    [6, 3, 4, 6, 2, 7, 4, 4, 6, 1],
    [2, 6, 2, 2, 7, 4, 3, 7, 7, 2],
    [5, 4, 1, 7, 3, 5, 5, 1, 7, 3],
    [4, 0, 3, 1, 5, 4, 3, 0, 0, 2],
    [2, 6, 1, 7, 3, 3, 7, 6, 5, 5]
]

y_lists = [
    [6, 1, 5, 3, 5, 4, 0, 4, 2, 3],
    [2, 0, 2, 4, 3, 7, 2, 1, 0, 2],
    [5, 1, 2, 1, 4, 1, 6, 7, 6, 1],
    [4, 4, 7, 0, 5, 1, 4, 4, 4, 6],
    [2, 0, 1, 0, 3, 6, 5, 3, 0, 5]
]

success = True

# Check for each list
for i in range(5):
    computed_y = compute_y(x_lists[i])
    if computed_y == y_lists[i]:
        print(f"Row {i}: Success")
    else:
        print(f"Row {i}: Failure")
        success = False

if success:
    print("All tests passed!")
else:
    print("Some tests failed.")