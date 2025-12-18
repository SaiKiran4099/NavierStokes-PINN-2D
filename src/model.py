import torch
import torch.nn as nn

class NavierStokes2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 30)
        self.layer1 = nn.Linear(30, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, 30)
        self.output = nn.Linear(30, 3)
        self.activation = nn.Tanh()

    def forward(self, inp):
        pass1 = self.activation(self.input(inp))
        pass2 = self.activation(self.layer1(pass1))
        pass3 = self.activation(self.layer2(pass2))
        pass4 = self.activation(self.layer3(pass3))
        pass5 = self.output(pass4)
        return pass5