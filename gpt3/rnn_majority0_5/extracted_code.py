from collections import Counter

# Define the function to compute list y
def compute_y(x):
    counts = Counter(x)
    y = [counts[ele] for ele in x]
    return y

# Define the lists x and y
list_x = [
    [0,5,4,4,0,5,4,2,4,5],
    [4,4,2,0,3,4,5,1,3,4],
    [1,2,1,5,5,1,5,3,1,1],
    [0,0,1,1,5,4,5,2,4,0],
    [2,0,1,3,3,5,5,4,3,5]
]

list_y_expected = [
    [0,0,0,4,0,0,4,4,4,4],
    [4,4,4,4,4,4,4,4,4,4],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [2,0,0,0,3,3,3,3,3,3]
]

# Compute and compare list y for the first 5 rows
for i in range(5):
    y = compute_y(list_x[i])
    if y == list_y_expected[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")