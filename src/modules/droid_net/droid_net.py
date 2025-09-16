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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src.modules.droid_net import ConvGRU, BasicEncoder, GradientClip


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2).contiguous()
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, kernel_size=(3, 3), padding=(1, 1))
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2, keepdim=False)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1).contiguous()
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data


def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)

    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
            nn.Softplus(),
        )

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, kernel_size=(1, 1), padding=(0, 0))
        )

    def forward(self, net, ii):
        # net: [1, n_edges, 128, 45, 80], ii_updated_first128_cnet_feature
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, sorted=True, return_inverse=True)
        net = self.relu(self.conv1(net))
        net =net.view(batch, num, 128, ht, wd)

        # Averages features from multiple factors that belong to the same frame
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd) # [1, n_unique_frames of ii!, 128, 45, 80]
        
        net = self.relu(self.conv2(net)) #[n_unique_frames, 128, 45, 80]
        eta = self.eta(net).view(batch, -1, ht, wd) # [1, n_unique_frames, 45, 80]
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd) #[1, n_unique_frames, 128->576, 45, 80]

        return 0.01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3+1)**2 # 196

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, kernel_size=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(7, 7), padding=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
        )

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, ii_first128_cnet_feature, ii_second128_cnet_feature, corr, flow=None, ii=None, jj=None):
        """
        This function can be used for both single frame optical flow estimation with last 3 inputs as None,
        It also can be used for batch
        ii_first128_cnet_feature: [1, n_edges, 128, ht(45), wd(80)], feature from ii
        ii_second128_cnet_feature: [1, n_edges, 128, ht(45), wd(80)], feature from ii
        corr: [1, n_edges, 196, ht(45), wd(80)], correlation features
        flow: [1, n_edges, 4, 45, 80], pixel motion in ii, and pixel motion in projection
        """
        batch, num, ch, ht, wd = ii_first128_cnet_feature.shape # [b, n, 128, 45, 80]
        device = ii_first128_cnet_feature.device

        # [b, n, 4, 45, 80]
        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=device)

        out_dim = (batch, num, -1, ht, wd) #[b, n, -1, 45, 80], -1 means auto deduce the dimension
        print(out_dim)

        # Reshape tensors for batch processing
        ii_first128_cnet_feature = ii_first128_cnet_feature.view(batch*num, -1, ht, wd) #[1*n_edges, 128, 45, 80]
        ii_second128_cnet_feature = ii_second128_cnet_feature.view(batch*num, -1, ht, wd) # [1*n_edges, 128, 45, 80]
        corr = corr.view(batch*num, -1, ht, wd) # [1*n_edges, 196, 45, 80]
        flow = flow.view(batch*num, -1, ht, wd) # [1*n_edges, 4, 45, 80]

        # Encode, pure cnn
        corr = self.corr_encoder(corr)  # [1*n_edges, 196->128, 45, 80]
        flow = self.flow_encoder(flow)  # [1*n_edges, 4->64, 45, 80]
        
        # [1*n_edges, 128, 45, 80]
        ii_updated_first128_cnet_feature = self.gru(ii_first128_cnet_feature, ii_second128_cnet_feature, corr, flow)
        # [1, n_edges, 2, 45, 80]
        motion_delta = self.delta(ii_updated_first128_cnet_feature).view(*out_dim)
        # [1, n_edges, 2, 45, 80]
        confidence_weight = self.weight(ii_updated_first128_cnet_feature).view(*out_dim)

        # Reshape[1, n_edges, ht, wd, 2] to [1, n_edges, ht, wd, 2] and keep only x,y components
        motion_delta = motion_delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        # [1, n_edges, ht, wd, 2]
        confidence_weight = confidence_weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        # [1, n_edges, 128, 45, 80]
        ii_updated_first128_cnet_feature = ii_updated_first128_cnet_feature.view(*out_dim)

        if ii is not None:
            # eta: Depth update magnitude for each pixel, [n_unique_frames of ii!, 45, 80]
            # upmask: Upsampling mask for depth refinement, [1, n_unique_frames_in_ii, 128->576, 45, 80]
            eta, upmask = self.agg(ii_updated_first128_cnet_feature, ii.to(device))
            return ii_updated_first128_cnet_feature, motion_delta, confidence_weight, eta, upmask
        else:
            return ii_updated_first128_cnet_feature, motion_delta, confidence_weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(out_dim=128, norm_fn='instance') #[b, n, 3, h, w] ->  [b, n, 128, h1/8, w1/8]
        self.cnet = BasicEncoder(out_dim=256, norm_fn='none') #[b, n, 3, h, w] ->  [b, n, 256, h1/8, w1/8]
        self.update = UpdateModule()

