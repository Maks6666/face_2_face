from torch import nn
import torch.nn.functional as F


class SkipConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if bnorm == True:
            self.bnorm1 = nn.BatchNorm2d(out_channels)
        else:
            self.bnorm1 = None

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)

        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        if self.bnorm1:
            out = self.bnorm1(out)

        out += residual
        out = F.relu(out)
        return out



class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        self.add_con = nn.Sequential()

        if in_channels != out_channels:
            self.add_con = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        if pool == True:
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        else:
            self.pool = None

    def forward(self, x):
        out = self.conv1(x)
        add_out = self.add_con(x)

        out = F.relu(out)

        out = self.conv2(out)
        out += add_out

        out = F.relu(out)

        if self.pool:
            out = self.pool(out)

        return out




