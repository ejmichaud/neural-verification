def compute_y_from_x(x_list):
    y_list = []
    length = len(x_list)
    for i in range(length):
        prev_x = x_list[i-1] if i > 0 else 0
        next_x = x_list[i+1] if i < length - 1 else 0
        current_x = x_list[i]

        # Placeholder for the actual decision logic. This will need to be
        # replaced with the actual logic you discover from observing patterns
        # in how y is derived from x.
        # For now, let's "guess" that y element becomes 1 when any of the
        # surrounding x elements was 1, otherwise 0.
        new_y = int(prev_x or current_x or next_x) # This rule is just an assumption!

        y_list.append(new_y)
    return y_list

# Define a function to test the success of our computed y against given y
def test_y_computation(pairs):
    for x, expected_y in pairs:
        computed_y = compute_y_from_x(x)
        if computed_y == expected_y:
            print('Success')
        else:
            print('Failure')

# List of pairs of (list x, list y) for the first five rows
pairs = [
    ([0,1,0,0,0,1,0,0,0,1],   [0,1,1,0,0,1,1,0,0,1]),
    ([0,0,0,0,1,0,1,1,1,0],   [0,0,0,0,1,1,1,0,0,1]),
    ([1,0,1,1,1,1,1,1,1,1],   [1,1,1,0,0,0,0,0,0,0]),
    ([0,0,1,1,1,0,1,0,0,0],   [0,0,1,0,0,1,1,1,0,0]),
    ([0,0,1,1,1,1,1,0,1,1],   [0,0,1,0,0,0,0,1,1,0])
]

test_y_computation(pairs)