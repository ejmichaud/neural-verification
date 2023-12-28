# Define the lists
x = [[0, 1],[0, 0],[0, 1],[0, 0],[0, 1],[0, 0],[0, 0],[1, 0],[1, 1],[1, 0]]
y_expected = [0,0,0,0,0,0,0,0,1,0]
y_actual = []

# Compute list y from list x
for i in range(5):
    y_actual.append(x[i][1])

# Check if the output matches list y
if y_actual == y_expected:
    print('Success')
else:
    print('Failure')