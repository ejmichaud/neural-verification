import random

def shuffle_lists_consistently(list1, list2, seed=42):
    combined = list(zip(list1, list2))
    random.Random(seed).shuffle(combined)
    list1[:], list2[:] = zip(*combined)

# Example usage:
list_a = [1, 2, 3, 4, 5]
list_b = ['a', 'b', 'c', 'd', 'e']

shuffle_lists_consistently(list_a, list_b)

print("List A:", list_a)
print("List B:", list_b)
