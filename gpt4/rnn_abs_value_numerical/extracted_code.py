# Define the lists x and y
lists_x = [
    [42, -33, -24, -86, -74, 35, -80, 24, 50, 13],
    [78, -86, -90, 54, -69, 72, 15, -5, 67, 6],
    [49, -24, -27, 11, -1, -87, 41, 69, -13, -81],
    [72, -20, -25, 29, 33, 64, 39, 76, -68, -90],
    [86, -78, 77, 19, 7, 23, 43, -6, -7, -23]
]

lists_y = [
    [42, 33, 24, 86, 74, 35, 80, 24, 50, 13],
    [78, 86, 90, 54, 69, 72, 15, 5, 67, 6],
    [49, 24, 27, 11, 1, 87, 41, 69, 13, 81],
    [72, 20, 25, 29, 33, 64, 39, 76, 68, 90],
    [86, 78, 77, 19, 7, 23, 43, 6, 7, 23]
]

# Function to compute y from x
def compute_y_from_x(x):
    return [abs(xi) for xi in x]

# Main program logic
def main():
    for index, x in enumerate(lists_x):
        # Compute y from x
        computed_y = compute_y_from_x(x)
        # Check if the computed y matches the provided y
        if computed_y == lists_y[index]:
            print(f"Row {index+1}: Success")
        else:
            print(f"Row {index+1}: Failure")

# Run the main function
main()