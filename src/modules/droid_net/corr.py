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
import torch.nn.functional as F

import droid_backends


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume, coords)
        ctx.radius = radius

        corr, = droid_backends.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = droid_backends.corr_index_backward(volume, coords, grad_output, ctx.radius)

        return grad_volume, None, None


class CorrBlock:
    """
    Multi-scale correlation block for visual correspondence matching.
    
    This class builds a correlation pyramid between two feature maps and provides
    efficient sampling of correlation features at different scales. It's used for
    estimating visual correspondences between consecutive frames in SLAM.
    
    The correlation pyramid enables coarse-to-fine matching:
    - Level 0: Full resolution correlation
    - Level 1: 1/2 resolution correlation  
    - Level 2: 1/4 resolution correlation
    - Level 3: 1/8 resolution correlation
    """
    
    def __init__(self, fmap1, fmap2, num_levels=4, radius=3):
        """
        Initialize correlation block with two feature maps.
        
        Args:
            fmap1: previous frame feature map shape [b, n, 128, 45, 80]
            fmap2: current frame feature map shape [b, n, 128, 45, 80]
            num_levels: Number of pyramid levels (default: 4)
            radius: Sampling radius for correlation features (default: 3)
        """
        self.num_levels = num_levels  # Number of pyramid levels
        self.radius = radius          # Sampling radius (3x3 neighborhood)

        #[
        # [b*n, 45, 80, 45, 80]
        # [b*n, 45, 80, 22, 40]
        # [b*n, 45, 80, 11, 20]
        # [b*n, 45, 80, 5, 10]
        #]
        self.corr_pyramid = []        # Store correlation pyramid

        # Step 1: Compute all-pairs correlation between feature maps
        corr = CorrBlock.corr(fmap1, fmap2) # [b, n, 45, 80, 45, 80]

        # Extract dimensions from correlation volume
        batch, num, h1, w1, h2, w2 = corr.shape # [b, n, 45, 80, 45, 80]
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2) # [b*n*45*80, 1, 45, 80]

        # Step 2: Build multi-scale correlation pyramid
        for i in range(self.num_levels):
            # Store current level of correlation pyramid
            # Shape: [b*n, h1, w1, h2//2**i, w2//2**i]
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i)
            )
            # Downsample for next level, [b*n*h1*w1, 1, h, w] -> [b*n*h1*w1, 1, h//2, w//2]
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)

    def __call__(self, coords):
        """
        Sample correlation features at specified coordinates.
        
        This method samples correlation features from the multi-scale pyramid
        at the given pixel coordinates. For each coordinate, it samples a
        radius x radius neighborhood from each pyramid level.
        
        Args:
            coords: Coordinate tensor shape [batch, num, ht, wd, 2]
                   where each pixel has (x, y) coordinates
                   
        Returns:
            torch.Tensor: Correlation features shape [batch, num, 4*radius^2, ht, wd]
                         where 4*radius^2 = 4*(2*3+1)^2 = 196 features per pixel
                         (49 features from each of 4 pyramid levels)
        """
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape # [b, n, 45, 80, 2]
        

        coords = coords.permute(0, 1, 4, 2, 3) # [b, n, 2, 45, 80]
        coords = coords.contiguous().view(batch*num, 2, ht, wd) # [b*n, 2, 45, 80]

        # Sample correlation features from each pyramid level
        # corr_pyramid:
        #[
        # [b*n, 45, 80, 45, 80]
        # [b*n, 45, 80, 22, 40]
        # [b*n, 45, 80, 11, 20]
        # [b*n, 45, 80, 5, 10]
        #]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            print(corr.shape)
            
            out_pyramid.append(corr.view(batch, num, -1, ht, wd)) # [b, n, 7*7, 45, 80] from each level

        # Concatenate features from all pyramid levels
        # Final shape: [batch, num, 4*radius^2, ht, wd] = [batch, num, 196, ht, wd]
        return torch.cat(out_pyramid, dim=2)

    def cat(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], dim=0)

        return self

    def __getitem__(self, index):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]

        return self

    @staticmethod
    def corr(fmap1, fmap2):
        """
        Compute all-pairs correlation between two feature maps.
        
        This method computes the dot product (correlation) between every pixel
        in fmap1 and every pixel in fmap2. The result is a 4D correlation volume
        that indicates how well each pixel in fmap1 matches each pixel in fmap2.
        
        Args:
            fmap1: First feature map shape [batch, num, 128, ht, wd] - previous frame
            fmap2: Second feature map shape [batch, num, 128, ht, wd] - current frame
            
        Returns:
            torch.Tensor: Correlation volume shape [batch, num, ht, wd, ht, wd]
                         where corr[b,n,i,j,k,l] = similarity between pixel (i,j) in fmap1
                         and pixel (k,l) in fmap2
        """
        # Extract dimensions from feature maps
        # [1, 1, 128, 45 (h/8), 80 (w/8)]
        batch, num, dim, ht, wd = fmap1.shape
        
        # Reshape to [1, 128, 45*80], and normalize by 4.0
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0

        # Compute all-pairs correlation using matrix multiplication
        # fmap1.transpose(1, 2): [1, 45*80, 128]
        # fmap2: [1, 128, 45*80]
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2) # [1, 45*80, 45*80]

        # Reshape to correlation volume: [1, 1, 45, 80, 45, 80]
        return corr.view(batch, num, ht, wd, ht, wd)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = droid_backends.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            droid_backends.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)

        return fmap1_grad, fmap2_grad, coords_grad, None


class AltCorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W) / 4.0

        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H//2**i, W//2**i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, kernel_size=2, stride=2)

    def corr_fn(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5) # [B, N, S, H, W, 2]

        corr_list = []
        for i in range(self.num_levels):
            fmap1_i = self.pyramid[0][:, ii] # [B, N, H, W, C]
            fmap2_i = self.pyramid[i][:, jj] # [B, N, H//2**i, W//2**i, C]

            coords_i = (coords / 2 ** i).reshape(B*N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B*N, ) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B*N, ) + fmap2_i.shape[2:])

            corr = CorrLayer.apply(fmap1_i.float(), fmap2_i.float(), coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)  # [B, N, (2r+1)^2, H, W, S]
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2) # [B, N, (2r+1)^2 * num_levels, H, W, S]

        return corr

    def __call__(self, coords, ii, jj):
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        # [B, N, (2r+1)^2 * num_levels, H, W, S]
        corr = self.corr_fn(coords, ii, jj)

        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()
