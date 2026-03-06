import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten()
        )

        conv_out = 32*3*3

        self.fc = nn.Sequential(
            nn.Linear(conv_out + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()  # action 범위 0~1
        )

    def forward(self, heightmap, features):
        h = self.cnn(heightmap)
        x = torch.cat([h, features], dim=1)
        return self.fc(x)