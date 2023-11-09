import numpy as np
import pandas as pd

def compute_all_roots():
    n_values = np.arange(1, 1000001)
    k_values = np.arange(2, 11)

    all_results = np.empty((len(n_values), len(k_values), 2))

    for i, k in enumerate(k_values):
        roots = np.round(n_values ** (1/k))
        all_results[:, i, 0] = f"{n_values},{k}"
        # all_results[:, i, 1] = k
        all_results[:, i, 1] = roots

    results_reshaped = all_results.reshape(-1, 2)
    return results_reshaped

all_data = compute_all_roots()

# To see some of the results:
for idx in range(10):
    print(all_data[idx])
    
# choose 75% of the results to be in the train set
# df = pd.DataFrame(all_data.astype(int), columns=["n", "k", "closest_root"])
df = pd.DataFrame(all_data, columns=["input", "closest_root"])
train_df = df.sample(frac=0.75)
test_df = df.drop(train_df.index)

# Save the dataframe to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
