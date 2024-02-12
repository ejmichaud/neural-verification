from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Provided list x and list y for the first five rows.
list_x = [
    [[1, 2], [4, 5], [4, 3], [5, 5], [0, 0], [0, 5], [6, 4], [1, 2], [6, 6], [2, 3]],
    [[1, 0], [0, 3], [3, 5], [1, 1], [2, 2], [2, 5], [0, 0], [5, 0], [4, 3], [6, 1]],
    [[0, 1], [5, 5], [6, 4], [0, 2], [5, 5], [1, 2], [5, 5], [1, 4], [6, 5], [3, 0]],
    [[5, 0], [2, 5], [5, 2], [4, 2], [6, 0], [4, 4], [3, 5], [0, 5], [3, 2], [6, 3]],
    [[0, 1], [6, 3], [4, 2], [6, 5], [4, 5], [5, 4], [1, 5], [5, 1], [4, 3], [4, 6]]
]
list_y = [
    [3, 2, 1, 4, 1, 5, 3, 4, 5, 6],
    [1, 3, 1, 3, 4, 0, 1, 5, 0, 1],
    [1, 3, 4, 3, 3, 4, 3, 6, 4, 4],
    [5, 0, 1, 0, 0, 2, 2, 6, 5, 2],
    [1, 2, 0, 5, 3, 3, 0, 0, 1, 4]
]

# Initialize the random forest regressor
model = RandomForestRegressor()

# Train the model using the provided lists for the first five rows.
X_train = np.array(list_x).reshape(-1, 2)  # Reshape list_x to two columns.
y_train = np.array(list_y).flatten()  # Flatten list_y to a single list.

# Fit the model.
model.fit(X_train, y_train)

# Predict using the trained model on the training data.
y_pred = model.predict(X_train)

# Check if predicted y matches with list y.
y_pred_rounded = np.round(y_pred).astype(int)
success = np.array_equal(y_pred_rounded, y_train)

# Print 'Success' if they match or 'Failure' if they don't.
print('Success' if success else 'Failure')