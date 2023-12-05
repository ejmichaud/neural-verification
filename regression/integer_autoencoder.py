import numpy as np
import torch
import os
from neural_verification import *
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.linear_model import LinearRegression

def normalize_basis(basis):
    '''Normalize unit cell basis'''
    D = basis.shape[1]
    basis = copy.deepcopy(basis)
    ii = 0
    while True and ii < 10:
        projs = []
        for i in range(D):
            proj = np.round(np.sum(basis*basis[[i]],axis=1)/np.linalg.norm(basis[i])**2)
            proj[i] = 0
            basis -= proj[:,np.newaxis] * basis[[i]]
            projs.append(proj)
        projs = np.array(projs)
        if np.sum(np.abs(projs)) == 0:
            break
        ii += 1
    basis *= (-1)**(np.sum(basis, axis=1)<0)[:,np.newaxis]
    return basis

def GCDautoencode(hidden, thres=1e-1, fraction=0.02): 
    '''
    Input: hidden states which are secretely a lattice X
    Output: Manifest integer lattice Z, bias b, and basis A. We want |X - AZ + b| small and Z close to integers
    two hyperparameters as input: 
    thres: a threshold parameter, smaller means more strict
    fraction: the algorithm works the best when the lattice does not have a too large size. So we would select points in the middle of the lattice.
    '''
    
    def subset(arr, criterion=None, batch_size = 32, seed=0):
        '''if arr contains too many samples, computation can be expensive.
        Choosing a subset of samples if usually enough for lattice detection.'''
        np.random.seed(seed)
        if criterion != None:
            arr = arr[np.where(criterion(arr))[0]]
        batch_id = np.random.choice(arr.shape[0], batch_size)
        return copy.deepcopy(arr[batch_id])

    def check_integer_arr(arr, thres=1e-1):
        '''Check an array how close to integers'''
        non_integer = np.abs(arr - np.round(arr)) > thres
        all_integer = np.sum(non_integer) == 0
        return all_integer


    def get_vol_arr(x):
        '''Given n points in R^D, return all possible volumes of D+1 points'''
        num = x.shape[0]
        D = x.shape[1]
        groups = np.array(list(itertools.combinations(x, D+1)))
        groups = groups[:,1:,:] - groups[:,[0],:]
        vols = np.abs(np.linalg.det(np.array(groups)))
        return np.array(vols)


    def GCD_2num_v(a, b, va, vb, thres=1e-1):
        '''This is a subroutine of GCD_arr.
        Find GCD of two numbers a, b via recursive subtraction. Transfer the same computations to two associated vectors va and vb. '''
        while True:
            temp = a
            a = b
            b = temp

            temp = va
            va = vb
            vb = temp

            proj = np.round(a/b)
            a = a - proj * b
            va = va - proj * vb
            if np.abs(a) < thres:
                break
        return np.abs(b), vb

    def GCD_arr(arr, thres=1e-1):
        '''Given a group of points, find unit cell basis'''
        D = arr.shape[1]
        vol_arr = get_vol_arr(arr)
        ii = 0
        arr_len = arr.shape[0]
        while arr.shape[0] >= D+2:
            va = arr[[0]]; vb = arr[[1]]; v3 = arr[2:D+1]
            a = np.linalg.det(np.concatenate([va, v3], axis=0))
            b = np.linalg.det(np.concatenate([vb, v3], axis=0))

            if b == 0 or a == 0:
                if b == 0:
                    arr = np.delete(arr, 1, 0)
                if a == 0:
                    arr = np.delete(arr, 0, 0)
                continue
            gcd, vbp = GCD_2num_v(a, b, va, vb, thres=thres)
            flag = check_integer_arr(vol_arr/gcd, thres=thres)
            if flag == True:
                break
            else:
                arr = arr[1:] # remove v1
                va = arr[0]; vb = arr[1]; vc = arr[2];
                arr[0] = vb; arr[1] = vc; arr[2] = vbp;
        return np.concatenate([vbp, v3], axis=0)

    def criterion(fraction=0.05):
        '''criterion to preprocess data points. Only select points in the middle of the lattice.'''
        def criterion_(hidden):
            D = hidden.shape[1]
            num = hidden.shape[0]
            #fraction = 0.01
            selection = np.ones(hidden.shape[0], dtype=int)
            for i in range(D):
                lim = np.sort(hidden[:,i])[int((0.5-fraction/2)*num):int((0.5+fraction/2)*num)]
                selection *= (hidden[:,i] < lim[-1]) * (hidden[:,i] > lim[0])
            if np.sum(selection) == 0:
                return np.ones(hidden.shape[0], dtype=int)
            return np.array(selection, dtype=bool)
        return criterion_
    
    # the dimension of hidden states
    D = hidden.shape[1]
    # preprocess data
    hidden_batch = subset(hidden, criterion=criterion(fraction=fraction))
    # get an estimate of the bias term, and shift hidden states by the bias term
    b = copy.deepcopy(hidden_batch[[np.argmin(np.sum(hidden_batch, axis=1))]])
    hidden_batch_shift = hidden_batch - b
    # A: basis vectors
    A = GCD_arr(copy.deepcopy(hidden_batch_shift), thres=thres)
    A = normalize_basis(A)
    
    # Z: integer lattice
    Z = np.matmul(hidden - b, np.linalg.inv(A))
    # translate integer lattice to be non-negative
    min_value = np.min(Z, axis=0)
    Z = Z - min_value[np.newaxis,:]
    b += np.sum(A * min_value[np.newaxis,:], axis=0)
    
    # correct b: Az + b = hidden
    error = np.matmul(Z, A) + b - hidden
    b -= np.mean(error, axis=0)

    return A,b,Z
# metrics to evaulate the quality of integer lattice

        # Metric 1: rounding error (never used in code; Max just wants to save it)
def rounding_error(Z):
    '''how close are Z to integers? Simply calculating the rounding error'''
    return np.mean(np.abs(Z - np.round(Z)))


# Metric 2: description length, which is simply addition of model complexity and data complexity
def description_length(X, A, b, Z, eps=1e-8):

    def model_complexity(Z):
        '''after rounding Z to integers, how many bits are needed to store Z?'''
        return np.mean(np.log2(1+np.abs(np.round(Z))))

    def data_complexity(X, A, b, Z, eps=1e-8):
        '''after rounding Z to integers, how large is the error = X - (A*Z + b)'''
        error = X - (np.matmul(np.round(Z), A) + b)
        return np.mean(np.log2(1+np.abs(error)/eps))

    model_c = model_complexity(Z)
    data_c = data_complexity(X, A, b, Z, eps=eps)
    dl = model_c + data_c
    if np.isnan(dl):
        dl = 1e8
    return dl
def integer_autoencoder_quality(X, A, b, Z, eps=1e-8):
    return description_length(X, A, b, Z, eps=eps), rounding_error(Z)

def LinRNNautoencode(X_last, X_last2, inputs_last): 
    '''
    X_last is the hidden states at the last position, 
    X_last2 is the hidden states at the second to last position,
    inputs_last is the inputs at the last position
    '''
    
    '''
    Determine linear regression from (h, x) to h.
    Wh: hidden to hidden, Wx: input to hidden
    '''
    X = np.concatenate([inputs_last, X_last2], axis=1)
    y = X_last
    reg = LinearRegression().fit(X, y)
    Wh = reg.coef_[:,1:]
    Wx = reg.coef_[:,[0]]
    
    '''
    determine basis vectors for the lattice
    '''
    x0s = []
    norms = []
    x0 = Wx
    hidden_dim = X_last.shape[1]
    for i in range(10):
        x0s.append(copy.deepcopy(x0))
        norms.append(np.linalg.norm(x0))
        x0 = np.matmul(Wh, x0)

    x0s = np.array(x0s)[:,:,0]
    norms = np.array(norms)
    A = x0s[np.argsort(norms)[::-1][:hidden_dim]]
    A = normalize_basis(A)
    
    '''
    determine a shift vector (move a hidden state to origin)
    choose the hidden state to be the one that minimize its coordinate sum (one can choose other conventions too)
    '''
    b = X_last[np.argmin(np.sum(X_last, axis=1))]
    Z_last = np.matmul(X_last - b, np.linalg.inv(A))
    Z_last2 = np.matmul(X_last2 - b, np.linalg.inv(A))
    
    
    # translate integer lattice to be non-negative
    min_value = np.min(Z_last, axis=0)
    Z_last = Z_last - min_value[np.newaxis,:]
    Z_last2 = Z_last2 - min_value[np.newaxis,:]
    b += np.sum(A * min_value[np.newaxis,:], axis=0)

    '''
    basis: basis vectors for the lattice
    b: global shift of the lattice
    Z_last: integer lattice for the last hidden states
    Z_last2: integer lattice for the second to last hidden states
    '''
    return A, b, Z_last, Z_last2

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device=torch.device("cpu")

    data = torch.load("../tasks/rnn_identity_numerical/data.pt")
    config = torch.load("../tasks/rnn_identity_numerical/model_config.pt", map_location=torch.device('cpu'))
    model = GeneralRNN(config,device=torch.device('cpu') )
    model.load_state_dict(torch.load("../tasks/rnn_identity_numerical/model_perfect.pt",map_location=torch.device('cpu')))
    
    def get_hidden(model,x):
         #x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)
        hidden = torch.zeros(batch_size, model.hidden_dim).to(device)
        outs = []
        for i in range(seq_length):
            out, hidden = model.forward(x[:,i,:], hidden)
            if i == seq_length-2:
                hidden_second_last = hidden.clone()
            outs.append(out)
        return torch.stack(outs).permute(1,0,2), hidden.cpu().detach().numpy(), hidden_second_last.cpu().detach().numpy()
    
    _, hidden, hidden2 = get_hidden(model,data[0].unsqueeze(dim=2))
    
    inputs_last = data[0][:,[-1]].cpu().detach().numpy()
    
    # LinRNNautoencode

    A, b, Z, _ = LinRNNautoencode(hidden, hidden2, inputs_last)

    print(description_length(hidden, A, b, Z))
    
    # GCDautoencode
    # try different hyperparameters for GCDautoencode

    fractions = 10 ** np.linspace(-3,0,num=10)
    thresholds = 10 ** np.linspace(-3,0,num=10)

    F, T = np.meshgrid(fractions, thresholds)
    hyperparams = np.transpose(np.array([F.reshape(-1,), T.reshape(-1,)]))

    dls = []
    for i in range(100):
        A, b, Z = GCDautoencode(hidden, thres=hyperparams[i][0], fraction=hyperparams[i][1])
        dls.append(description_length(hidden, A, b, Z))


    optimal_hp = hyperparams[np.argmin(dls)]
    A, b, Z = GCDautoencode(hidden, thres=optimal_hp[0], fraction=optimal_hp[1])
    print(description_length(hidden, A, b, Z))