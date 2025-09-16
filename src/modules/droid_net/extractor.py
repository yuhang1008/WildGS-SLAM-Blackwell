# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if stride > 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if stride > 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if stride > 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride > 1:
                self.norm3 = nn.Sequential()
        else:
            raise TypeError(norm_fn)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


DIM = 32


class BasicEncoder(nn.Module):
    def __init__(self, out_dim, norm_fn='batch'):
        super(BasicEncoder, self).__init__()
        self.out_dim = out_dim
        self.norm_fn = norm_fn

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM) # normal

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        else:
            raise TypeError(self.norm_fn)

        self.conv1 = nn.Conv2d(3, DIM, 7, 2, 3) # 3: input channels, DIM: output channels, 7: kernel size, 2: stride, 3: padding
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        self.conv2 = nn.Conv2d(4*DIM, out_dim, kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = [layer1, layer2]

        self.in_planes = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x) # b*n , DIM , h1 , w1 -> b*n , DIM , h1/2 , w1/2
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x) # b*n , DIM , h1/2 , w1/2 -> b*n , DIM , h1/2 , w1/2
        x = self.layer2(x) # b*n , DIM , h1/2 , w1/2 -> b*n , 2*DIM , h1/4 , w1/4
        x = self.layer3(x) # b*n , 2*DIM , h1/4 , w1/4 -> b*n , 4*DIM , h1/8 , w1/8

        x = self.conv2(x) # b*n , 4*DIM , h1/8 , w1/8 -> b*n , out_dim , h1/8 , w1/8

        _, c2, h2, w2 = x.shape
        x = x.view(b, n, c2, h2, w2) # b, n, out_dim, h1/8, w1/8

        return x