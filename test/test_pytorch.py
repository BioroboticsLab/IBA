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
