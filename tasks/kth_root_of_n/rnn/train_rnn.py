import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tasks.kth_root_of_n.rnn.rnn_model import RNN



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

def load_data(mode='train'):
    # load data from a csv file into a pandas dataframe
    data = np.loadtxt('./{}.csv'.format(mode), delimiter=',', dtype='float64', skiprows=1)
    inputs = data[:,:2]
    labels = data[:,2]
    inputs_ = np.transpose(np.array(inputs)[:,:,np.newaxis],(0, 2, 1)).astype('float')
    labels_ = np.array(labels)[:,np.newaxis,np.newaxis].astype('float')
    return inputs_, labels_

inputs_train, labels_train = load_data(mode='train')
inputs_test, labels_test = load_data(mode='test')
print(inputs_train.shape, labels_train.shape, inputs_test.shape, labels_test.shape)
print(inputs_train[0])
inputs_train = torch.tensor(inputs_train, dtype=torch.float64, requires_grad=True).to(device)
labels_train = torch.tensor(labels_train, dtype=torch.float64, requires_grad=True).to(device)
inputs_test = torch.tensor(inputs_test, dtype=torch.float64, requires_grad=True).to(device)
labels_test = torch.tensor(labels_test, dtype=torch.float64, requires_grad=True).to(device)

print(inputs_train.shape, labels_train.shape, inputs_test.shape, labels_test.shape)
def l1(model):
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_reg
    

model = RNN(hidden_dim=2, device=device).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
steps = 1001
log = 1
lamb = 0e-4

dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs_train, labels_train), batch_size=10000, shuffle=False)

for step in range(steps):
    
    for i, (inputs_train, labels_train) in enumerate(dataloader):
        optimizer.zero_grad()
        
        pred_train = model(inputs_train)
        loss_train = torch.mean((pred_train-labels_train)**2)
        acc_train = torch.sum(torch.abs(pred_train-labels_train) < 0.25)/len(labels_train)

        reg = l1(model)
        loss = loss_train + lamb * reg
        
        loss.backward()
        optimizer.step()
    
    if step % log == 0:
        pred_test = model(inputs_test)
        loss_test = torch.mean((pred_test-labels_test)**2)
        acc_test = torch.sum(torch.abs(pred_test-labels_test) < 0.25)/len(labels_test)
        
        print("step = %d | train loss: %.2e | test loss %.2e | train acc: %.2e | test acc: %.2e | reg: %.2e "%(step, loss_train.cpu().detach().numpy(), loss_test.cpu().detach().numpy(), acc_train.cpu().detach().numpy(), acc_test.cpu().detach().numpy(), reg.cpu().detach().numpy()))

test_print = inputs_test[:10]
print(model(test_print))
print(labels_test[:10])

torch.save(model.state_dict(), './model')
