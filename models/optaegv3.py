import torch
import torch.nn as nn
import torch.nn.functional as F


class OptAEGV3(nn.Module):
    """
    OptAEGV3 - Optimized activation function module
    This variant can reach 98.2% accuracy on MNIST with only 702 parameters.
    Performance is quite stable. It is derived from transformer concepts.
    """
    def __init__(self):
        super().__init__()
        self.vx = nn.Parameter(torch.zeros(1, 1, 1))
        self.vy = nn.Parameter(torch.ones(1, 1, 1))
        self.wx = nn.Parameter(torch.zeros(1, 1, 1))
        self.wy = nn.Parameter(torch.ones(1, 1, 1))
        self.afactor = nn.Parameter(torch.zeros(1, 1))
        self.mfactor = nn.Parameter(torch.ones(1, 1))

    def flow(self, dx, dy, data):
        return data * (1 + dy) + dx

    def forward(self, data):
        shape = data.size()
        data = data.flatten(1)
        data = data - data.mean()
        data = data / data.std()

        b = shape[0]
        v = self.flow(self.vx, self.vy, data.view(b, -1, 1))
        w = self.flow(self.wx, self.wy, data.view(b, -1, 1))

        dx = self.afactor * torch.sum(v * torch.sigmoid(w), dim=-1)
        dy = self.mfactor * torch.tanh(data)
        data = self.flow(dx, dy, data)

        return data.view(*shape)


class TinyMNIST(nn.Module):
    """
    Tiny MNIST model using OptAEGV3 activation
    Extremely efficient model with custom activation function
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.lnon0 = OptAEGV3()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon1 = OptAEGV3()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        self.lnon2 = OptAEGV3()
        self.fc = nn.Linear(4 * 3 * 3, 10, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lnon0(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.lnon1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.lnon2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
