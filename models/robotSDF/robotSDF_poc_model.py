import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotSDF(nn.Module):
    def __init__(self) -> None:
        super(RobotSDF, self).__init__()

        self.layer1 = nn.Linear(7, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, q):

        q = F.relu(self.layer1(q))
        q = F.relu(self.layer2(q))
        outs = self.layer3(q)

        return outs