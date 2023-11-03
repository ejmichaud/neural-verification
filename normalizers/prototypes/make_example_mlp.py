import torch
import torch.nn as nn
import numpy as np

from neural_verification import MLP


# Make an example MLP and save it in sample_model.pt
model = MLP(in_dim=10, out_dim=15, width=30, depth=4)
torch.save(model.state_dict(), 'sample_model.pt')
