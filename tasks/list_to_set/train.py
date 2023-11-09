import torch.nn as nn
import numpy as np 
import torch 
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
class DuplicateRemoverRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(DuplicateRemoverRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        x = self.embeddings(x)
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)
        out = self.softmax(out)
        return out

def preprocess_data(sequences, targets, vocab_size=10):
    # Pad sequences for consistent length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    # One-hot encode targets
    targets_with_zero = [torch.cat((tar, torch.zeros(1, dtype=tar.dtype))) for tar in targets]
    padded_targets = pad_sequence(targets_with_zero, batch_first=True, padding_value=0)
    
    target_tensors = torch.zeros(len(targets), padded_sequences.shape[1], vocab_size, dtype=torch.float32)
    for i, target in enumerate(padded_targets):
        # Ensure target is a LongTensor
        target_long = target.type(torch.long)
        target_tensors[i] = torch.nn.functional.one_hot(target_long, num_classes=vocab_size)
    return padded_sequences, target_tensors

# Load data
data_train = np.loadtxt("data_train.txt", dtype=str)

sequences = [torch.tensor(list(map(int, sequence)), dtype=torch.long) for sequence in data_train[:,0]]
targets = [torch.tensor(list(map(int, target)), dtype=torch.long) for target in data_train[:,1]]

# Process data
sequences, targets = preprocess_data(sequences, targets)
sequences = sequences.cuda()
targets = targets.cuda()

data_test = np.loadtxt("data_test.txt", dtype=str)
test_sequences = [torch.tensor(list(map(int, sequence)), dtype=torch.long) for sequence in data_test[:,0]]
test_targets = [torch.tensor(list(map(int, target)), dtype=torch.long) for target in data_test[:,1]]
test_sequences, test_targets = preprocess_data(test_sequences, test_targets)
test_sequences = test_sequences.cuda()
test_targets = test_targets.cuda()

vocab_size = 10  # for digits 1-9 and 0 as padding
embedding_dim = 10
hidden_dim = 128
batch_size = 32

# Create the model
model = DuplicateRemoverRNN(vocab_size, embedding_dim, hidden_dim, batch_size).cuda()

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def train_and_evaluate(model, train_X, train_y, test_X, test_y, optimizer, loss_function, epochs=10):
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for i in tqdm(range(0, train_X.size(0), batch_size)):
            batch_X = train_X[i:i+batch_size]
            batch_y = train_y[i:i+batch_size].argmax(2)  # Converting from one-hot to indices
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_function(output.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_X)

        # Testing
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for i in range(0, test_X.size(0), batch_size):
                batch_X = test_X[i:i+batch_size]
                batch_y = test_y[i:i+batch_size].argmax(2)  # Converting from one-hot to indices
                
                output = model(batch_X)
                loss = loss_function(output.view(-1, vocab_size), batch_y.view(-1))
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_X)

        print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_test_loss:.4f}')

train_and_evaluate(model, sequences, targets, test_sequences, test_targets, optimizer, loss_function)
 
def predict_sequence(model, seq, vocab_size):
    model.eval()
    with torch.no_grad():
        padded_seq = pad_sequence([seq], batch_first=True, padding_value=0)
        prediction = model(padded_seq)
        _, predicted_indices = prediction.max(2)
        return torch.unique(predicted_indices[0], sorted=False, return_inverse=False)

# Test the prediction function
new_sequence = torch.tensor([1, 2, 2, 3, 1, 4, 4, 5]).cuda()
print(predict_sequence(model, new_sequence, vocab_size).cpu().numpy())
