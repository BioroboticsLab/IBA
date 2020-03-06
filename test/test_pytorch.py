import torch
import torch.nn as nn
import torch.nn.functional as F
from IBA.pytorch import IBA


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_pytorch():
    net = Net()
    iba = IBA(net.conv2)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad(), iba.interrupt_execution(), iba.enable_estimation():
        net(x)

    def generator():
        for i in range(3):
            yield torch.randn(2, 3, 32, 32), None

    iba.estimate(net, generator(), progbar=True)

    out = net(x)
    with iba.supress_information():
        out_with_noise = net(x)

    assert (out != out_with_noise).any()

    img = torch.randn(1, 3, 32, 32)
    iba.heatmap(img, lambda x: -torch.log_softmax(net(x), 1)[:, 0].mean())

    iba.detach()

    # no influence after detach
    with iba.supress_information():
        out_with_noise = net(x)

    assert (out == out_with_noise).any()
