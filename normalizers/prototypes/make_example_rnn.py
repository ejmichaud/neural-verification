import torch
import torch.nn as nn
import numpy as np

from neural_verification import RNN


# Make an example MLP and save it in sample_rnn.pt
model = RNN(30, input_dim=2, output_dim=1)
torch.save(model.state_dict(), 'sample_rnn.pt')
