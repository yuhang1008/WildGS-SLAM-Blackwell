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

from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, device):
    y, x = torch.meshgrid(
        torch.arange(ht).to(device).float(),  # y coordinates: [0, 1, 2, ..., ht-1]
        torch.arange(wd).to(device).float(),  # x coordinates: [0, 1, 2, ..., wd-1]
        indexing="ij")

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),indexing="ij")

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """
    Map points from frame ii to frame jj
    
    This function transforms 3D points visible in frame ii to their corresponding 
    pixel locations in frame jj. It's the core operation for visual odometry and
    SLAM - given camera poses and depth, where do points from one frame appear in another?
    
    Pipeline:
    1. Take 2D pixels + depth from frame ii → 3D world points
    2. Transform 3D points from frame ii's coordinate system to frame jj's coordinate system  
    3. Project 3D points onto 2D pixels in frame jj
    
    WHY JACOBIANS?
    This function is used in Bundle Adjustment optimization loops where camera poses and 
    depths are iteratively refined. The Jacobians tell us:
    - Ji: How much the projected pixels change when we adjust the SOURCE frame pose (ii)
    - Jj: How much the projected pixels change when we adjust the TARGET frame pose (jj)  
    - Jz: How much the projected pixels change when we adjust the depth values
    
    This allows the optimizer to compute gradients and update poses/depths to minimize
    reprojection error: ||observed_pixels - projected_pixels||²
    
    Args:
        poses: Camera poses [B, N, 7] (position + quaternion for each frame)
        depths: Depth maps [B, N, H, W] for each frame
        intrinsics: Camera intrinsics [B, N, 4] (fx, fy, cx, cy) for each frame
        ii: Source frame indices [M] - which frames to transform FROM
        jj: Target frame indices [M] - which frames to transform TO
        jacobian: Whether to compute derivatives for optimization
        return_depth: Whether to return depth information
    
    Returns:
        x1: Projected pixel coordinates in frame jj [B, M, H, W, 2]
        valid: Validity mask for points [B, M, H, W, 1]
        (Ji, Jj, Jz): Jacobians w.r.t. poses and depths (if jacobian=True)
    """

    # ============================================================================
    # STEP 1: INVERSE PROJECTION - 2D pixels + depth → 3D world points
    # ============================================================================
    # Convert 2D pixel coordinates + depth from frame ii into 3D world coordinates
    # This "unprojects" the image back into 3D space using the camera model
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    # X0: 3D points in frame ii's coordinate system [B, M, H, W, 4] (X, Y, Z, depth)
    # Jz: Jacobian w.r.t. depth (if jacobian=True)
    
    # ============================================================================
    # STEP 2: COORDINATE TRANSFORMATION - Transform between camera coordinate systems
    # ============================================================================
    # Compute the transformation from frame ii's coordinate system to frame jj's
    # Gij = T_jj * T_ii^(-1) = "transform from ii to jj"
    Gij = poses[:,jj] * poses[:,ii].inv()
    # Gij: SE(3) transformation matrix from frame ii to frame jj

    # Handle special case: if ii == jj (same frame), set to identity transformation
    # This prevents numerical issues when transforming a frame to itself
    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    
    # Apply the transformation: X1 = Gij * X0
    # Transform 3D points from frame ii's coordinate system to frame jj's coordinate system
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    # X1: 3D points in frame jj's coordinate system [B, M, H, W, 4]
    # Ja: Jacobian w.r.t. pose transformation (if jacobian=True)
    
    # ============================================================================
    # STEP 3: PROJECTION - 3D world points → 2D pixel coordinates
    # ============================================================================
    # Project the transformed 3D points onto the 2D image plane of frame jj
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)
    # x1: 2D pixel coordinates in frame jj [B, M, H, W, 2] (u, v)
    # Jp: Jacobian w.r.t. projection (if jacobian=True)

    # ============================================================================
    # STEP 4: VALIDITY CHECK - Filter out invalid points
    # ============================================================================
    # Exclude points that are too close to the camera (invalid depth)
    # Points must be valid in BOTH source and target frames
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)
    # valid: Binary mask [B, M, H, W, 1] indicating which points are valid

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),indexing="ij")

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid

