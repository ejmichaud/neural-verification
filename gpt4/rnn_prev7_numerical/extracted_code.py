# Define the lists of x and their corresponding lists of y.
data_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70, 39, 86, 99, 15, 84, 78, 8],
    [66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84, 53, 48, 9, 83, 7, 58, 91],
    [14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65, 89, 41, 45, 61, 39, 61, 64],
]

data_y = [
    [0, 0, 0, 0, 0, 0, 0, 42, 67, 76, 14, 26, 35, 20, 24, 50, 13, 78, 14, 10],
    [0, 0, 0, 0, 0, 0, 0, 49, 76, 73, 11, 99, 13, 41, 69, 87, 19, 72, 80, 75],
    [0, 0, 0, 0, 0, 0, 0, 86, 22, 77, 19, 7, 23, 43, 94, 93, 77, 70, 9, 70],
    [0, 0, 0, 0, 0, 0, 0, 66, 30, 40, 60, 70, 61, 23, 20, 11, 61, 77, 89, 84],
    [0, 0, 0, 0, 0, 0, 0, 14, 91, 36, 3, 82, 90, 89, 28, 55, 33, 27, 47, 65],
]

# Define a function to transform list_x to list_y.
def transform_list(list_x):
    return [0]*7 + list_x[7:]

# Iterate over the first five rows, transform list_x to list_y and compare with the provided list_y.
for i in range(5):
    computed_list_y = transform_list(data_x[i])
    if computed_list_y == data_y[i]:
        print(f"Row {i+1}: Success")
    else:
        print(f"Row {i+1}: Failure")