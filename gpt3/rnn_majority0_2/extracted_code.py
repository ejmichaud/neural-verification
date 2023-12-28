# Define the lists
x = [
    [0,2,1,1,0,2,1,2,1,2],
    [1,1,2,0,0,1,2,1,0,1],
    [1,2,1,2,2,1,2,0,1,1],
    [0,0,1,1,2,1,2,2,1,0],
    [2,0,1,0,0,2,2,1,0,2]
]

y_expected = [
    [0,0,0,1,0,0,1,1,1,1],
    [1,1,1,1,0,1,1,1,1,1],
    [1,1,1,1,2,1,2,2,1,1],
    [0,0,0,0,0,1,1,1,1,1],
    [2,0,0,0,0,0,0,0,0,0]
]

# Compute list y from list x for the first 5 rows
y_computed = []
for row in x[:5]:
    y_row = [row.count(i) for i in row]
    y_computed.append(y_row)

# Check if the output matches list y and print 'Success' or 'Failure'
if y_computed == y_expected:
    print('Success')
else:
    print('Failure')