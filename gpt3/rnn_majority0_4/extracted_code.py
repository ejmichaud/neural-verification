from collections import Counter

def compute_y(x):
    counts = Counter(x)
    y = [counts[i] for i in x]
    return y

# Input data
data = [
    ([2, 2, 1, 4, 1, 0, 0, 4, 0, 3], [2, 2, 2, 2, 1, 1, 0, 0, 0, 0]),
    ([3, 4, 0, 4, 1, 2, 0, 0, 2, 1], [3, 3, 0, 4, 4, 4, 0, 0, 0, 0]),
    ([4, 1, 3, 1, 4, 3, 1, 4, 2, 4], [4, 1, 1, 1, 1, 1, 1, 1, 1, 4]),
    ([2, 0, 0, 4, 3, 4, 4, 1, 2, 0], [2, 0, 0, 0, 0, 0, 4, 4, 4, 0]),
    ([1, 2, 2, 4, 2, 3, 3, 4, 3, 2], [1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
]

# Compute y and check for match
for x, expected_y in data:
    computed_y = compute_y(x)
    if computed_y == expected_y:
        print('Success')
    else:
        print('Failure')