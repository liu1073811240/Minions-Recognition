from torch import nn,optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )#batch*64*112*112

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )#batch*128*56*56

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )#batch*256*28*28

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )#batch*512*14*14

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )#batch*256*7*7

        self.conv56 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )#batch*128*3*3

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )#batch*64*1*1

        self.fcn = nn.Sequential(
            nn.Linear(in_features=64*1*1,out_features=5),

        )

    def forward(self, x):
        # print(x.type())
        # x = torch.FloatTensor(x)
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y5 = self.conv56(y5)
        y6 = self.conv6(y5)

        y6 = y6.reshape(y6.size(0), -1)
        output = self.fcn(y6)  # [N, 5]

        output1 = F.relu(output[:, :4])  # [N, 4]

        output2 = torch.sigmoid(output[:, 4:])  # [N, 1]

        return output1, output2


