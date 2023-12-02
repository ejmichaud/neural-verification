import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
parser.add_argument('-d', type=str, dest='dataset_name', required=True)
parsed_args = parser.parse_args()
dataset_name = parsed_args.dataset_name

#class Seq2SeqModel(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size):
#        super(Seq2SeqModel, self).__init__()
#        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
#        self.output_layer = nn.Linear(hidden_size, output_size)
#
#    def forward(self, x):
#        rnn_out, _ = self.rnn(x)
#        output = self.output_layer(rnn_out)
#        return output

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outs = []
        hiddens = [torch.zeros((x.size()[1], self.hidden_size))]
        for token in range(x.size()[0]):
            hiddens.append(torch.tanh(self.in2hidden(x[token,:,:]) + self.hidden2hidden(hiddens[-1])))
#            hiddens.append(self.in2hidden(x[token,:,:]) + self.hidden2hidden(hiddens[-1]))
            outs.append(self.hidden2out(hiddens[-1]))
        hiddens = torch.stack(hiddens[1:], dim=0)
        output = torch.stack(outs, dim=0)
        return hiddens, output

class AlgorithmicDatasets:
    def __init__(self):
        self.dataset_functions = {
            "identity": self.generate_identity_data,
            "shift": self.generate_shift_data,
#            "cumulative_min": self.generate_cumulative_min_data,
#            "cumulative_max": self.generate_cumulative_max_data,
            "binary_addition": self.generate_binary_addition_data,
            "cumulative_sum_mod_5": self.generate_cumulative_sum_mod_5_data,
            "cumulative_sum_mod_19": self.generate_cumulative_sum_mod_19_data,
            # Add more dataset functions here
        }
        
        self.dataset_shapes = {
            "identity": {
                "input_size": 3,
                "hidden_size": 2,
                "output_size": 3,
            },
            "shift": {
                "input_size": 1,
                "hidden_size": 5,
                "output_size": 2,
            },
#            "cumulative_min": {
#                "input_size": 1,
#                "hidden_size": 10,
#                "output_size": 1,
#            },
#            "cumulative_max": {
#                "input_size": 1,
#                "hidden_size": 10,
#                "output_size": 1,
#            },
            "binary_addition": {
                "input_size": 2,
                "hidden_size": 40,
                "output_size": 2,
            },
            "cumulative_sum_mod_5": {
                "input_size": 5,
                "hidden_size": 20,
                "output_size": 5,
            },
            "cumulative_sum_mod_19": {
                "input_size": 19,
                "hidden_size": 30,
                "output_size": 19,
            },
            # Define shapes for other datasets
        }
        self.dataset_hyperparameters = {
            "identity": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.003,
                "num_epochs": 100,
                "batch_size": 32,
            },
            "shift": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 1000,
                "batch_size": 32,
            },
            "cumulative_min": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 1000,
                "batch_size": 32,
            },
            "cumulative_max": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 1000,
                "batch_size": 32,
            },
            "binary_addition": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 1000,
                "batch_size": 32,
            },
            "cumulative_sum_mod_5": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 10000,
                "batch_size": 32,
            },
            "cumulative_sum_mod_19": {
                "dataset_size": 100,
                "sequence_length": 100,
                "learning_rate": 0.001,
                "num_epochs": 10000,
                "batch_size": 32,
            },
        }

    def generate_identity_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        test_samples = int(dataset_size * (1 - train_ratio))
        train_samples = dataset_size - test_samples
        
        train_out = torch.randint(0, 3, (sequence_length, train_samples)).long()
        test_out = torch.randint(0, 3, (sequence_length, test_samples)).long()

        train_in = torch.nn.functional.one_hot(train_out, num_classes=3).float()
        test_in = torch.nn.functional.one_hot(test_out, num_classes=3).float()
        
        return train_in, train_out, test_in, test_out

    def generate_shift_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        test_samples = int(dataset_size * (1 - train_ratio))
        train_samples = dataset_size - test_samples
        
        train_in = torch.randint(0, 2, (sequence_length, train_samples, 1)).float()
        test_in = torch.randint(0, 2, (sequence_length, test_samples, 1)).float()
        
        train_out = torch.cat([torch.zeros((1, train_samples)).float(), train_in[:-1,:,0]], dim=0).long()
        test_out = torch.cat([torch.zeros((1, test_samples)).float(), test_in[:-1,:,0]], dim=0).long()
        
        return train_in, train_out, test_in, test_out

#    def generate_cumulative_min_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
#        test_samples = int(dataset_size * (1 - train_ratio))
#        train_samples = dataset_size - test_samples
#        
#        train_in = torch.randn(sequence_length, train_samples, 1).float()
#        test_in = torch.randn(sequence_length, test_samples, 1).float()
#        
#        train_out = torch.cummin(train_in, 1)[0].float()
#        test_out = torch.cummin(test_in, 1)[0].float()
#        
#        return train_in, train_out, test_in, test_out
#
#    def generate_cumulative_max_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
#        test_samples = int(dataset_size * (1 - train_ratio))
#        train_samples = dataset_size - test_samples
#        
#        train_in = torch.randn(sequence_length, train_samples, 1).float()
#        test_in = torch.randn(sequence_length, test_samples, 1).float()
#        
#        train_out = torch.cummax(train_in, 1)[0].float()
#        test_out = torch.cummax(test_in, 1)[0].float()
#        
#        return train_in, train_out, test_in, test_out

    def generate_binary_addition_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        test_samples = int(dataset_size * (1 - train_ratio))
        train_samples = dataset_size - test_samples
        
        def make_split(samples):
            ints1 = [int(torch.randint(0, 2**(sequence_length-1), ())) for _ in range(samples)]
            ints2 = [int(torch.randint(0, 2**(sequence_length-1), ())) for _ in range(samples)]

            num1 = [list(map(int, bin(x)[2:].zfill(sequence_length))) for x in ints1]
            num2 = [list(map(int, bin(x)[2:].zfill(sequence_length))) for x in ints2]
            sums = [list(map(int, bin(x+y)[2:].zfill(sequence_length))) for x, y in zip(ints1, ints2)]

            summands = torch.transpose(torch.stack([torch.tensor(num1), torch.tensor(num2)], dim=2), 0, 1).float().flip(0)
            sums = torch.tensor(np.array(sums).T).long().flip(0)

            return summands, sums

        train_in, train_out = make_split(train_samples)
        test_in, test_out = make_split(test_samples)
        
        return train_in, train_out, test_in, test_out

    def generate_cumulative_sum_mod_19_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        return self.generate_cumulative_sum_mod_n_data(19, dataset_size=dataset_size, sequence_length=sequence_length, train_ratio=train_ratio)

    def generate_cumulative_sum_mod_5_data(self, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        return self.generate_cumulative_sum_mod_n_data(5, dataset_size=dataset_size, sequence_length=sequence_length, train_ratio=train_ratio)

    def generate_cumulative_sum_mod_n_data(self, n, dataset_size=1000, sequence_length=10, train_ratio=0.8):
        test_samples = int(dataset_size * (1 - train_ratio))
        train_samples = dataset_size - test_samples
        
        def make_split(samples):
            in_nums = torch.randint(0, n, (sequence_length, samples))
            out_nums = torch.cumsum(in_nums, dim=0) % n

            def convert(x):
                return torch.nn.functional.one_hot(x, num_classes=n).float()

            return convert(in_nums), out_nums

        train_in, train_out = make_split(train_samples)
        test_in, test_out = make_split(train_samples)
        
        return train_in, train_out.long(), test_in, test_out.long()

    def get_dataset(self, dataset_name, **kwargs):
        if dataset_name in self.dataset_functions:
            return self.dataset_functions[dataset_name](**kwargs)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not available in the dataset functions.")

    def get_dataset_shapes(self, dataset_name):
        if dataset_name in self.dataset_shapes:
            return self.dataset_shapes[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' does not have defined shapes.")

    def get_dataset_hyperparameters(self, dataset_name):
        if dataset_name in self.dataset_hyperparameters:
            return self.dataset_hyperparameters[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' does not have defined hyperparameters.")

if __name__ == "__main__":
    dataset_manager = AlgorithmicDatasets()

    # Define hyperparameters
    hyperparameters = dataset_manager.get_dataset_hyperparameters(dataset_name)
    dataset_size = hyperparameters["dataset_size"]
    sequence_length = hyperparameters["sequence_length"]
    learning_rate = hyperparameters["learning_rate"]
    num_epochs = hyperparameters["num_epochs"]
    batch_size = hyperparameters["batch_size"]

    train_in, train_out, test_in, test_out = dataset_manager.get_dataset(dataset_name, dataset_size=dataset_size, sequence_length=sequence_length, train_ratio=0.8)
    print(f"Train Input Shape: {train_in.shape}")
    print(f"Train Output Shape: {train_out.shape}")
    print(f"Test Input Shape: {test_in.shape}")
    print(f"Test Output Shape: {test_out.shape}")

    # Initialize the model
    dataset_shapes = dataset_manager.get_dataset_shapes(dataset_name)
    input_size = dataset_shapes["input_size"]
    hidden_size = dataset_shapes["hidden_size"]
    output_size = dataset_shapes["output_size"]
    model = Seq2SeqModel(input_size, hidden_size, output_size)
    print(model)

    # Create a DataLoader for training data
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=False)  # SHUFFLE=TRUE BREAKS THE IN/OUT CORRESPONDENCE
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function (you may need to adjust this depending on the specific problem)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for in_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            _, output = model(in_batch)

            # Calculate loss (you may need to adjust the loss depending on the problem)
    #        print(output.size())
    #        print(target_batch.size())
            loss = criterion(output.transpose(1, 2), target_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    print("Training complete!")

    model.eval()
    total_test_loss = 0

    for in_batch, target_batch in test_loader:
        with torch.no_grad():
            _, output = model(in_batch)
            test_loss = criterion(output.transpose(1, 2), target_batch)
            total_test_loss += test_loss.item()

    print(f"Test Loss: {total_test_loss / len(test_loader)}")

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/" + dataset_name + "_model.pt")
