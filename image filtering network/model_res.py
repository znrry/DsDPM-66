import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class CNNModelWithResidual(nn.Module):
    def __init__(self):
        super(CNNModelWithResidual, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.residual1 = ResidualBlock(16, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.residual2 = ResidualBlock(32, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.residual3 = ResidualBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.residual4 = ResidualBlock(128, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.residual1(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.residual2(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.residual3(x)
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.residual4(x)
        #print(x.shape)
        x = x.view(-1, 128 * 16 * 16)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def plot_training_loss(self, losses):
        plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved at {file_path}")

# Creating Models
model = CNNModelWithResidual()

# Printed Model Structures
print(model)