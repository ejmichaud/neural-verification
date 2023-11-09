import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import RNN
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
### Preparation ####

# set random seed
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# set precision and device
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

### load dataset ###
vocab_size = 10
def load_data(mode='train'):
    data = np.loadtxt('./data_{}.txt'.format(mode), dtype='str')
    inputs = data[:,0]
    labels = data[:,1]

    def strs2mat(strings):
        num = strings.shape[0]
        mat = []
        for i in range(num):
            mat.append([*strings[i]])
        return mat

    inputs = [torch.tensor(list(map(int, sequence)), dtype=torch.long) for sequence in inputs]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    inputs_ = torch.zeros(len(inputs), padded_inputs.shape[1], vocab_size, dtype=torch.float32)
    for i, sequence in enumerate(padded_inputs):
        # Ensure target is a LongTensor
        sequence = sequence.type(torch.long)
        inputs_[i] = torch.nn.functional.one_hot(sequence, num_classes=vocab_size)
    print(inputs_.shape)
    labels_ = torch.tensor( [int(elem) for elem in labels])

    return inputs_, labels_

inputs_train, labels_train = load_data(mode='train')
print(inputs_train[0])
print(labels_train[0])
inputs_test, labels_test = load_data(mode='test')

inputs_train = torch.tensor(inputs_train, dtype=torch.float64, requires_grad=True).to(device)
labels_train = torch.tensor(labels_train, dtype=torch.float64, requires_grad=True).to(device)
inputs_test = torch.tensor(inputs_test, dtype=torch.float64, requires_grad=True).to(device)
labels_test = torch.tensor(labels_test, dtype=torch.float64, requires_grad=True).to(device)


def l1(model):
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_reg
    

model = RNN(hidden_dim=10,input_dim=10, device=device).to(device)


# ### Training ###

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
steps = 1001
log = 200
lamb = 0e-4

batch_size = 32
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(inputs_train, labels_train)
test_dataset = TensorDataset(inputs_test, labels_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10 

for epoch in range(num_epochs):
    model.train() 
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        pred_train = model(inputs)
        loss_train = torch.mean((pred_train-labels)**2)
        acc_train = torch.mean(((pred_train - 0.5)*(labels_train - 0.5) > 0).long().float())
        
        reg = l1(model)
        loss = loss_train + lamb * reg
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            if batch_idx == 0:
                pred_test = model(inputs)
                print(pred_test)
                print(labels)
                
            pred_test = model(inputs)
            loss_test = torch.mean((pred_test-labels)**2)
            acc_test = torch.mean(((pred_test - 0.5)*(labels_test - 0.5) > 0).long().float())
        
        
        
    print("epoch = %d | train loss: %.2e | test loss %.2e | train acc: %.2e | test acc: %.2e | reg: %.2e "%(epoch, loss_train.cpu().detach().numpy(), loss_test.cpu().detach().numpy(), acc_train.cpu().detach().numpy(), acc_test.cpu().detach().numpy(), reg.cpu().detach().numpy()))

torch.save(model.state_dict(), './model')

def predict_sequence(model, seq):
    model.eval()
    with torch.no_grad():
        padded_seq = pad_sequence([seq], batch_first=True, padding_value=0)
        prediction = model(padded_seq)
        return prediction.item()


new_sequence = torch.tensor([1, 2, 2, 3, 1, 4, 4, 5]).cuda()
print(predict_sequence(model, new_sequence).cpu().numpy())
