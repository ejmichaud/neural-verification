# Define the lists x and lists y as per the data provided for the first 5 rows
lists_x = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

lists_y = [
    [42, 109, 185, 157, 116, 75, 81, 79, 94, 87],
    [78, 92, 102, 78, 95, 157, 118, 182, 177, 168],
    [49, 125, 198, 160, 183, 123, 153, 123, 197, 175],
    [72, 152, 227, 184, 137, 126, 136, 179, 147, 118],
    [86, 108, 185, 118, 103, 49, 73, 160, 230, 264]
]

def compute_cumulative_sum(x):
    y = []
    cumulative_sum = 0
    for element in x:
        cumulative_sum += element
        y.append(cumulative_sum)
    return y

def check_success(computed_y, given_y):
    return computed_y == given_y

# Compute and check each list
for i in range(5):
    computed_y = compute_cumulative_sum(lists_x[i])
    if check_success(computed_y, lists_y[i]):
        print(f"Success for row {i+1}")
    else:
        print(f"Failure for row {i+1}")

# Alternative you can aggregate the results and print success or failure once

# Run the checks and success/failure outcome
all_success = True
for i in range(len(lists_x)):
    computed_y = compute_cumulative_sum(lists_x[i])
    if computed_y != lists_y[i]:
        all_success = False
        break

# Print the final outcome
if all_success:
    print("Success")
else:
    print("Failure")