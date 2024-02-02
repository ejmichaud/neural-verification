def compute_y_using_lookup(x_list, lookup):
    y_computed = [lookup[x] for x in x_list]
    return y_computed

# Let's first establish the mappings from the first row
x_values = [[2, 3, 0, 2, 2, 3, 0, 0, 2, 1],
            [2, 2, 2, 2, 3, 0, 3, 3, 3, 2],
            [1, 0, 1, 3, 3, 1, 1, 1, 3, 3],
            [0, 0, 3, 1, 1, 0, 3, 0, 0, 2],
            [2, 2, 1, 3, 3, 3, 3, 2, 1, 1]]

y_values = [[2, 1, 1, 3, 1, 0, 0, 0, 2, 3],
            [2, 0, 2, 0, 3, 3, 2, 1, 0, 2],
            [1, 1, 2, 1, 0, 1, 2, 3, 2, 1],
            [0, 0, 3, 0, 1, 1, 0, 0, 0, 2],
            [2, 0, 1, 0, 3, 2, 1, 3, 0, 1]]

# Derive a lookup mapping from the first row, however, there are multiple y's for some x's (not a function)
# so we can use the first occurrence as a trial.
x_to_y_mapping = {}
for x, y in zip(x_values[0], y_values[0]):
    x_to_y_mapping[x] = y

# Now let's test this derived mapping on the first five rows.
for i in range(5):
    y_computed = compute_y_using_lookup(x_values[i], x_to_y_mapping)
    if y_computed == y_values[i]:
        print(f"Row {i+1} Success:", y_computed)
    else:
        print(f"Row {i+1} Failure: computed {y_computed} but expected {y_values[i]}")