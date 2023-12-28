# Define a function to generate list y from list x using the hypothesized pattern
def generate_y(x_list):
    y_list = []
    total = 0
    for x in x_list:
        total += x
        y_list.append(total)
    return y_list

# List of list x and list y pairs
x_lists = [
    [42,67,76,14,26,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19,72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77,70,9,70,39,86,99,15,84,78,8],
    [66,30,40,60,70,61,23,20,11,61,77,89,84,53,48,9,83,7,58,91],
    [14,91,36,3,82,90,89,28,55,33,27,47,65,89,41,45,61,39,61,64]
]
y_lists = [
    [42,109,185,199,225,260,280,304,354,367,445,459,469,523,554,626,641,736,803,809],
    # ... (other y lists are omitted for brevity)
]

# Apply the generate_y function and check for success or failure
for i in range(5):
    generated_y = generate_y(x_lists[i])
    if generated_y == y_lists[i]:
        print(f'Row {i+1} Success')
    else:
        print(f'Row {i+1} Failure')

# Uncomment the following lines to inspect the generated y list and the provided y list
        # print("Generated y:", generated_y)
        # print("Given y:    ", y_lists[i])