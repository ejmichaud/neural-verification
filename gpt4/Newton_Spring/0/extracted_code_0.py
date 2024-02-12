# Function to calculate the cumulative sum list from list x
def cumulative_sum(x):
    y = []
    current_sum = 0
    for element in x:
        current_sum += element
        y.append(current_sum)
    return y

# Data for the first 5 rows provided
list_x = [
    [5, -8, -6, 9, 8, 7, 9, -5, -3, 4],
    [-3, 9, 10, 8, 5, 6, 10, 3, -1, 0],
    [-9, 4, -3, 7, 7, 9, -2, 5, 6, 6],
    [-1, 2, -3, -3, -5, -3, 1, 7, 3, 5],
    [4, 5, 9, 2, -4, 1, 4, 6, 2, -5]
]

list_y = [
    [5, -3, -14, -2, 20, 29, 18, -16, -37, -17],
    [-3, 6, 19, 21, 7, -8, -5, 6, 10, 4],
    [-9, -5, 1, 13, 19, 15, -6, -16, -4, 18],
    [-1, 1, -1, -5, -9, -7, 3, 17, 17, 5],
    [4, 9, 14, 7, -11, -17, -2, 21, 25, -1]
]

# Check if the output matches and print 'Success' or 'Failure'
success = True
for i in range(5):
    calculated_y = cumulative_sum(list_x[i])
    if calculated_y == list_y[i]:
        print(f"Row {i+1} Success")
    else:
        print(f"Row {i+1} Failure")
        success = False

if success:
    print("All checks passed successfully!")
else:
    print("There were failures in the output.")