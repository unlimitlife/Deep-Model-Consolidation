import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, tasks=10, task_size=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(32,64,3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64,64,3, stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.drop2 = nn.Dropout(p=0.5)
        self.classifier = nn.ModuleList(
                [nn.Linear(1024, task_size) for _ in range(tasks)])

    def forward(self, x, task):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier[task](x)
        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([fc(x) for fc in self.classifier], dim=-1)
        return x

def convnet(num_classes=100, **kwargs):
    model = ConvNet()
    return model
