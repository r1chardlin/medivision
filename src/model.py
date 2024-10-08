import torch.nn as nn
import torch.nn.functional as F

# v3 (version 3)
class ConvNet3(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_dim=14):
        super(ConvNet3, self).__init__()

        self.name = "v3"

        # 1024 * 1024 -> 1032 * 1032 (padding) -> 344 * 344 (convolution w/ stride=3) -> 86 * 86 (max pool w/ kernel_size=4)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=3, padding=4, bias=False)

        self.norm1 = nn.BatchNorm2d(num_features=hidden_channels)

        # 86 * 86 -> 88 * 88 (padding) -> 84 * 84 -> (convolution) -> 42 * 42 (max pool w/ kernel_size=2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(num_features=hidden_channels)

        # 42 * 42 -> 44 * 44 (padding) -> 40 * 40 (convolution) -> 8 * 8 (max pool w/ kernel_size=5)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=1, bias=False)

        self.norm3 = nn.BatchNorm2d(num_features=hidden_channels)

        self.fc = nn.Linear(8 * 8 * hidden_channels, output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4, stride=4)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 5, stride=5)

        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x

# v2 (version 2)
class ConvNet2(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_dim=14):
        super(ConvNet2, self).__init__()

        self.name = "v2"

        # 1024 * 1024 -> 1032 * 1032 (padding) -> 344 * 344 (convolution w/ stride=3) -> 86 * 86 (max pool w/ kernel_size=4)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=3, padding=4, bias=False)

        # 86 * 86 -> 88 * 88 (padding) -> 84 * 84 -> (convolution) -> 42 * 42 (max pool w/ kernel_size=2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=1, bias=False)

        # 42 * 42 -> 44 * 44 (padding) -> 40 * 40 (convolution) -> 8 * 8 (max pool w/ kernel_size=5)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=1, bias=False)

        self.fc = nn.Linear(8 * 8 * hidden_channels, output_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 4, stride=4)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 5, stride=5)

        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x

# v1
class ConvNet1(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_dim=14):
        super(ConvNet1, self).__init__()

        self.name = "v1"

        # 1024 * 1024 -> 1026 * 1026 (padding) -> 1024 * 1024 (convolution) -> 256 * 256 (max pool w/ kernel_size=4)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 256 * 256 -> 258 * 258 (padding) -> 256 * 256 -> (convolution) -> 64 * 64 (max pool w/ kernel_size=4)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 64 * 64 -> 68 * 68 (padding) -> 64 * 64 (convolution) -> 16 * 16 (max pool 4)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 16 * 16 -> 18 * 18 (padding) -> 16 * 16 (convolution) -> 8 * 8 (max pool w/ kernel_size=2)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc = nn.Linear(8 * 8 * hidden_channels, output_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 4, stride=4)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4, stride=4)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 4, stride=4)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, stride=2)
        
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)

        return x