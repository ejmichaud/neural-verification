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


