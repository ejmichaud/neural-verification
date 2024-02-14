# Given data
data = [
    ([6, 3, 12, 14, 10, 7, 12, 4, 6, 9], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]),
    ([2, 6, 10, 10, 7, 4, 3, 7, 7, 2], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]),
    ([5, 4, 1, 7, 11, 13, 5, 1, 15, 11], [0, 0, 0, 1, 0, 1, 1, 0, 1, 0]),
    ([4, 0, 11, 9, 5, 12, 11, 8, 0, 10], [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]),
    ([10, 14, 9, 15, 11, 11, 15, 14, 13, 13], [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]),
]

def even_odd_rule(x_values):
    return [0 if x % 2 == 0 else 1 for x in x_values]

def threshold_rule(x_values, threshold=10):
    return [0 if x < threshold else 1 for x in x_values]

def check_rule(rule_func, data):
    for x_values, expected_y in data:
        computed_y = rule_func(x_values)
        if computed_y == expected_y:
            print('Success')
        else:
            print('Failure', computed_y, 'does not match', expected_y)

# Test even-odd rule
print("Testing even-odd rule:")
check_rule(even_odd_rule, data)

# Test threshold rule
print("\nTesting threshold rule:")
check_rule(threshold_rule, data)