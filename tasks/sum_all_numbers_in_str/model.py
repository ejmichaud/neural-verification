import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, hidden_dim, input_dim=2, output_dim=1, device='cpu'):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wy = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        #self.act = lambda x: x
        self.device = device
    
    def forward(self, x):
        
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        outs = []

        for i in range(seq_length):
            hidden = self.act(self.Wh(hidden) + self.Wx(x[:,i,:]))
            out = self.Wy(hidden)
            outs.append(out)
        
        return torch.stack(outs).permute(1,0,2)