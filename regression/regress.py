import numpy as np
import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from neural_verification import *
import matplotlib.pyplot as plt
import itertools
import copy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import argparse
import pickle


# Create arugments for the script
parser = argparse.ArgumentParser(description='Specifications for RNN to Lattice to Regression')
parser.add_argument('--task', type=str, default='rnn_identity_numerical', help='task name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--regression', type=str, default='all methods', help='regression method')

args = parser.parse_args()
task = args.task
seed = args.seed
device = args.device
regression = args.regression

np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device(device)


data = torch.load(f"../tasks/{task}/data.pt", map_location=device)
output_dir = f"./tasks/{task}_hiddens"
hidden_last = torch.load(os.path.join(output_dir, 'hidden_last.pt'))
hidden_second_last = torch.load(os.path.join(output_dir, 'hidden_second_last.pt'))
D = hidden_last.shape[1]
thres = 1e-1


def subset(arr, criterion=None, batch_size = 32):
    '''if arr contains too many samples, computation can be expensive.
    Choosing a subset of samples if usually enough for lattice detection.'''
    if criterion != None:
        arr = arr[np.where(criterion(arr))[0]]
    batch_id = np.random.choice(arr.shape[0], batch_size)
    arr = arr[batch_id]
    return arr  


# GCD

def get_vol_arr(x):
    '''Given n points in R^D, return all possible volumes of D+1 points'''
    num = x.shape[0]
    groups = np.array(list(itertools.combinations(x, D+1)))
    groups = groups[:,1:,:] - groups[:,[0],:]
    vols = np.abs(np.linalg.det(np.array(groups)))
    return np.array(vols)


def GCD_arr(arr):
    '''Find the greatest common divisor (GCD) for an array'''
    vol_arr = get_vol_arr(arr)
    while True:
        va = arr[[0]]; vb = arr[[1]]; v3 = arr[2:D+1]
        a = np.linalg.det(np.concatenate([va, v3], axis=0))
        b = np.linalg.det(np.concatenate([vb, v3], axis=0))
        
        # should be more robust? Threshold?
        if b == 0 or a == 0:
            if b == 0:
                arr = np.delete(arr, 1, 0)
            if a == 0:
                arr = np.delete(arr, 0, 0)
            continue
        gcd, vbp = GCD_2num_v(a, b, va, vb)
        flag = check_integer_arr(vol_arr/gcd)
        if flag == True:
            break
        else:
            arr = arr[1:]
            arr[0] = vbp
    return np.concatenate([vbp, v3], axis=0)

def GCD_2num_v(a, b, va, vb):
    '''Find the greatest common divisor (GCD) for two number a, b;
    and apply the same rule to two vectors va and vb'''
    
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

def check_integer_arr(arr):
    """ 
    Check if all elements in an array are integers
    """
    non_integer = np.abs(arr - np.round(arr)) > thres
    all_integer = np.sum(non_integer) == 0
    return all_integer

def normalize_basis(basis):
    basis = copy.deepcopy(basis)
    ii = 0
    while True and ii < 5:
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



# shift lattice such that (0,0) is on lattice
hidden_batch = subset(hidden_last)
shift_id = np.argmin(np.sum(hidden_batch, axis=1))
shift = copy.deepcopy(hidden_batch[[shift_id]])
hidden_batch_shift = hidden_batch - shift
basis = GCD_arr(hidden_batch_shift)
basis_comp = normalize_basis(basis)



h_int_lattice = np.matmul(hidden_last - shift, np.linalg.inv(basis_comp))
h_last_int_lattice = np.matmul(hidden_second_last - shift, np.linalg.inv(basis_comp))
input_lattice = data[0][:,-1].cpu().detach().numpy()
output_lattice = data[1][:,-1].cpu().detach().numpy()


#<-------------------------possible fit methods below-------------------->
#rename integer_linear_regress
def linear(input, output, show_eqn=True):
    reg = LinearRegression().fit(input, output)
    score = reg.score(input, output)
    coeff = reg.coef_
    rounded_coeff = np.round(coeff)
    intercept = reg.intercept_
    new_out = input@ rounded_coeff + output
    if show_eqn:
        equation = "y = " + " + ".join([f"{coef}*x{idx}" for idx, coef in enumerate(coeff)]) + f" + {intercept}"
        print(f"Linear Equation: {equation}")
    return score, reg, None

# Modified polynomial fit function
def polynomial_fit(input, output, degree=2, show_eqn=True):
    poly = PolynomialFeatures(degree=degree)
    input_poly = poly.fit_transform(input)
    reg = LinearRegression().fit(input_poly, output)
    score = reg.score(input_poly, output)
    if show_eqn:
        features = poly.get_feature_names()
        equation_terms = [f"{coef}*{feature}" for coef, feature in zip(reg.coef_, features)]
        equation = "y = " + " + ".join(equation_terms) + f" + {reg.intercept_}"
        print(f"Polynomial Equation (degree {degree}): {equation}")
    return score, reg, poly.get_feature_names()
#<------------------------------------------------------------------------>


# methods = [linear, polynomial_fit]

# (h, x) => h
# (h_last_lattice, input_lattice) => h_lattice
# print("(h, x) => h")
hx = np.concatenate([np.round(h_last_int_lattice), np.round(input_lattice)[:,np.newaxis]], axis=1)
h_next = np.round(h_int_lattice)

best_score1, reg1, model_details1 = linear(hx, h_next, show_eqn=True)

# h => y
# h_lattice => output_lattice
# print("h => y")


best_score2, reg2, model_details2 = linear(np.round(h_int_lattice), output_lattice,  show_eqn=True)


if best_score1 >0.99 and best_score2>0.99:
    print(True)
    # Save lambda functions to text files
    print(f"Saved discovered hidden_to_hidden function to {output_dir}")
    with open(os.path.join(output_dir, 'hidden_to_hidden.pkl'), 'wb') as file:
        pickle.dump(reg1, file)
    print(f"Saved discovered hidden to out function to {output_dir}")
    with open(os.path.join(output_dir, 'hidden_to_output.pkl'), 'wb') as file:
        pickle.dump(reg2, file)
else:
    print(False)
    

