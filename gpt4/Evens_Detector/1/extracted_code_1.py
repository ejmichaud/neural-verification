# Define the function to compute y based on an element of x
def compute_y(x_element):
    return 1 if x_element % 2 == 0 else 0

# Define the input and expected output for the first 5 rows
x_lists = [
    [2, 7, 6, 4, 6, 5, 0, 4, 0, 3],
    [8, 4, 0, 4, 1, 2, 5, 5, 7, 6],
    [9, 6, 3, 1, 9, 3, 1, 9, 7, 9],
    [2, 0, 5, 9, 3, 4, 9, 6, 2, 0],
    [6, 2, 7, 9, 7, 3, 3, 4, 3, 7]
]

y_expected_lists = [
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0]
]

# Function to compare two lists
def compare_lists(list1, list2):
    return list1 == list2

# Calculate and compare the lists
for i in range(len(x_lists)):
    computed_y = [compute_y(x_element) for x_element in x_lists[i]]
    print(f"Row {i+1}: ", end="")
    if compare_lists(computed_y, y_expected_lists[i]):
        print("Success")
    else:
        print("Failure")