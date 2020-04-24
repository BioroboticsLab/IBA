# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from IBA.pytorch import IBA
from packaging import version


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
    if version.parse(torch.__version__) < version.parse("1.2.0"):
        iba = IBA()
        net.conv2 = nn.Sequential(net.conv2, iba)
    else:
        iba = IBA(net.conv2)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad(), iba.interrupt_execution(), iba.enable_estimation():
        net(x)

    def generator():
        for i in range(3):
            yield torch.randn(2, 3, 32, 32), None

    iba.estimate(net, generator(), progbar=True)

    out = net(x)
    with iba.restrict_flow():
        out_with_noise = net(x)

    assert (out != out_with_noise).any()

    img = torch.randn(1, 3, 32, 32)
    iba.analyze(img, lambda x: -torch.log_softmax(net(x), 1)[:, 0].mean())

    x = torch.randn(2, 3, 32, 32)
    out = net(x)

    if version.parse(torch.__version__) < version.parse("1.2.0"):
        with pytest.raises(ValueError):
            iba.detach()

        with iba.restrict_flow():
            out_with_noise = net(x)
        assert (out != out_with_noise).any()

    else:
        iba.detach()

        # no influence after detach
        with iba.restrict_flow():
            out_with_noise = net(x)

        assert (out == out_with_noise).all()

    with pytest.raises(ValueError):
        iba.detach()


def test_pytest_readme():
    # resembles the example in the readme
    # small changes to be fast and run on travis
    from IBA.pytorch import IBA, tensor_to_np_img
    from IBA.utils import plot_saliency_map, to_unit_interval

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    # Initialize some pre-trained model to analyze
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = Net()
    model.to(dev)

    # setup data loader
    val_set = CIFAR10(".", train=False, download=True, transform=ToTensor())
    val_loader = DataLoader(val_set, batch_size=50, shuffle=True, num_workers=4)

    # Add a Per-Sample Bottleneck at layer conv4_1
    if version.parse(torch.__version__) < version.parse("1.2.0"):
        iba = IBA()
        model.conv2 = nn.Sequential(model.conv2, iba)
    else:
        iba = IBA(model.conv2)

    # Estimate the mean and variance of the feature map at this layer.
    iba.estimate(model, val_loader, n_samples=100, progbar=True)

    # Explain class target for the given image
    img, target = val_set[0]
    saliency_map = iba.analyze(
        img.unsqueeze(0).to(dev),
        lambda x: -torch.log_softmax(model(x), dim=1)[:, target].mean(),
        beta=10)

    # display result
    np_img = to_unit_interval(tensor_to_np_img(img))
    plot_saliency_map(saliency_map, np_img)
