def compute_y(x):
    # Initialize y with zeros for the first 4 elements
    y = [0] * 4
    
    # Append the remaining elements of x to y, shifted right by 4 places
    y += x[:-4]
    
    return y

# Given data for the first five rows
x_values = [
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
    [49, 76, 73, 11, 99, 13, 41, 69, 87, 19],
    [72, 80, 75, 29, 33, 64, 39, 76, 32, 10],
    [86, 22, 77, 19, 7, 23, 43, 94, 93, 77]
]

y_values = [
    [0, 0, 0, 0, 42, 67, 76, 14, 26, 35],
    [0, 0, 0, 0, 78, 14, 10, 54, 31, 72],
    [0, 0, 0, 0, 49, 76, 73, 11, 99, 13],
    [0, 0, 0, 0, 72, 80, 75, 29, 33, 64],
    [0, 0, 0, 0, 86, 22, 77, 19, 7, 23]
]

success = True

for i in range(5):
    computed_y = compute_y(x_values[i])
    if computed_y != y_values[i]:
        print(f"Failure on row {i+1}")
        success = False
        break  # Stop checking further rows on first failure

if success:
    print("Success")