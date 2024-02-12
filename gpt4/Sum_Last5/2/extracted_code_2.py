# Define the lists of x
list_of_xs = [
    [42,67,76,14,26,35,20,24,50,13,78,14,10,54,31,72,15,95,67,6],
    [49,76,73,11,99,13,41,69,87,19,72,80,75,29,33,64,39,76,32,10],
    [86,22,77,19,7,23,43,94,93,77,70,9,70,39,86,99,15,84,78,8],
    [66,30,40,60,70,61,23,20,11,61,77,89,84,53,48,9,83,7,58,91],
    [14,91,36,3,82,90,89,28,55,33,27,47,65,89,41,45,61,39,61,64]
]

# Define the corresponding lists of y
list_of_ys = [
    [42,109,185,199,225,260,280,304,354,367,445,459,469,523,554,626,641,736,803,809],
    [49,125,198,209,308,321,362,431,518,537,609,689,764,793,826,890,929,1005,1037,1047],
    [86,108,185,204,211,234,277,371,464,541,611,620,690,729,815,914,929,1013,1091,1099],
    [66,96,136,196,266,327,350,370,381,442,519,608,692,745,793,802,885,892,950,1041],
    [14,105,141,144,226,316,405,433,488,521,548,595,660,749,790,835,896,935,996,1060]
]

# Function to compute y from x
def compute_y_from_x(x_list):
    y_list = []
    running_sum = 0
    for x in x_list:
        running_sum += x
        y_list.append(running_sum)
    return y_list

# Check if computed y matches the given y list
success = True
for i in range(len(list_of_xs)):
    computed_y = compute_y_from_x(list_of_xs[i])
    if computed_y == list_of_ys[i]:
        print(f"Row {i+1} - Success")
    else:
        print(f"Row {i+1} - Failure")
        success = False

# Print the overall result
if success:
    print("All computations matched list y - Global Success")
else:
    print("One or more computations did not match list y - Global Failure")