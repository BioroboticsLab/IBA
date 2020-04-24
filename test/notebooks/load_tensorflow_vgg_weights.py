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




import collections
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch.nn import Conv2d
from torchvision import transforms


class TensorflowVGGWeights:
    def __init__(self, device):
        self.device = device

        # load weights and patterns as tensors
        params_file = 'vgg_16_weights.npz'
        weights = self.load_params(params_file)
        
        # construct a VGG16 with these weights
        self.model = torchvision.models.vgg16(False).to(device)
        for i, p in enumerate(self.model.parameters()):
            p.data.copy_(weights[i])

    def load_params(self, filename):
        """ load parameters as np.ndarray from the file, and return them as a list of tensors on [device] """
        f = np.load(filename)
        weights = []
        transpose_idxs = [26, 28, 30]
        for i in range(32):
            if i in transpose_idxs:
                weights.append(torch.tensor(f['arr_%d' % i].T, device=self.device, requires_grad=False))
            else:
                weights.append(torch.tensor(f['arr_%d' % i], device=self.device, requires_grad=False))
        return weights


    def get_model(self):
        return self.model


class TensorflowTransform(object):
    def __init__(self):
        self.scale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.offset = np.array([103.939, 116.779, 123.68])[:, np.newaxis, np.newaxis]

    def __call__(self, raw_img):
        scaled_img = self.scale(raw_img)
        ret = np.array(scaled_img, dtype=np.float)
        # Channels first
        ret = ret.transpose((2, 0, 1))
        # Remove pixel-wise mean.
        # To BGR.
        ret = ret[::-1, :, :]
        ret -= self.offset
        ret = np.ascontiguousarray(ret)
        ret = torch.from_numpy(ret).float()
        return ret