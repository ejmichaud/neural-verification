# Provided lists of x
list_of_xs = [
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
]

# Expected lists of y
expected_list_of_ys = [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
]

# Function to compute y from x
def compute_y_from_x(x):
    return [i % 2 for i in range(len(x))]

# Main checking logic
success = True
for i in range(len(list_of_xs)):
    computed_y = compute_y_from_x(list_of_xs[i])
    if computed_y != expected_list_of_ys[i]:
        print(f"Failure on row {i+1}")
        success = False
        break

if success:
    print("Success")

# Run the program
print("Running the program...")
if __name__ == "__main__":
    main()