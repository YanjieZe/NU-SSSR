from torch import nn


class SRCNN2(nn.Module):
    """
    A deeper and larger SRCNN
    """
    def __init__(self, args):
        super(SRCNN2, self).__init__()
        num_channels = 3
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=9, padding=9 // 2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=9, padding=9 // 2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv5 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x