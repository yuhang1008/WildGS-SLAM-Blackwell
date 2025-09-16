import torch
import numpy as np

from src.modules.droid_net import CorrBlock, AltCorrBlock
import src.geom.projective_ops as pops
from copy import deepcopy


class FactorGraph:
    """
    Factor Graph for Visual-Inertial SLAM
    
    This class manages the factor graph structure used in SLAM (Simultaneous Localization and Mapping).
    A factor graph is a bipartite graph containing:
    - Variable nodes: camera poses and 3D points
    - Factor nodes: constraints between variables (e.g., reprojection constraints, motion constraints)
    
    The factor graph is mainly inherited from GO-SLAM and handles:
    - Adding/removing factors (edges) between frames
    - Managing correlation volumes for visual features
    - Performing bundle adjustment optimization
    - Handling inactive and bad factors
    """
    
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        """
        Initialize the Factor Graph
        
        Args:
            video: Video object containing camera poses, images, and features
            update_op: Update operator for the factor graph optimization
            device: Computing device (default: "cuda:0")
            corr_impl: Correlation implementation type ("volume" or "alt")
            max_factors: Maximum number of factors to maintain (-1 for unlimited)
        """
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # Image dimensions at 1/8 resolution (downscaled for efficiency)
        self.ht = ht = video.ht // self.video.down_scale
        self.wd = wd = video.wd // self.video.down_scale

        # Coordinate grid for pixel locations
        self.coords0 = pops.coords_grid(ht, wd, device=device)
        
        # Factor graph edge indices (frame i to frame j connections)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)  # source frame indices
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)  # target frame indices
        self.age = torch.as_tensor([], dtype=torch.long, device=device)  # age of each factor

        # Feature representations and correlation data
        self.corr = None  # correlation object! corr.corr_pyramid[level] is a list of tensors, [batch, n_edge, 196, ht/2^level, wd/2^level]
        self.net = None  # first128_cnet_feature of ii, from video, [1, n_edge, 128, ht (45), wd (80)]
        self.inp = None  # second128_cnet_feature of ii, from video, [1, n_edge, 128, ht (45), wd (80)]
        self.damping = 1e-6 * torch.ones_like(self.video.disps)  # damping factors for optimization

        # Reprojection targets and weights for bundle adjustment
        # [1 (batch), n_deges, ht, wd, 2]
        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)  # target pixel locations
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)  # confidence weights

        # Inactive factors indices (temporarily removed but stored for potential reuse)
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)  # bad factors indices to avoid
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """
        Remove duplicate edges to avoid redundant factors in the graph
        
        Args:
            ii: Source frame indices
            jj: Target frame indices
            
        Returns:
            Filtered ii, jj tensors with duplicates removed
        """
        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        
        # Create set of existing edges (both active and inactive)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        # Keep only edges that don't already exist
        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        """
        Print all edges in the factor graph with their weights for debugging
        """
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        # Sort edges by source frame index
        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        # Calculate average weight for each edge
        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        
        # Print edge information: (source_frame, target_frame, weight)
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """
        Remove bad edges based on confidence threshold and temporal distance
        
        Bad edges are those with:
        - Temporal distance > 2 frames AND
        - Low confidence weight < 0.001
        """
        # Calculate confidence as mean weight across spatial dimensions
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        
        # Identify bad edges: far apart in time AND low confidence
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        # Move bad edges to bad factors list
        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        
        # Remove bad factors from active graph
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        """
        Clear all edges and reset the factor graph to empty state
        Used for complete graph reset
        """
        # Clear active factors
        self.ii = None
        self.jj = None
        self.age = None
        self.corr = None
        self.damping = None
        self.net = None
        self.inp = None
        self.target = None 
        self.weight = None
        
        # Clear inactive and bad factors
        self.ii_inac = None
        self.jj_inac = None
        self.ii_bad = None
        self.jj_bad = None
        self.target_inac = None
        self.weight_inac = None

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def add_factors(self, ii, jj, remove=False):
        """
        Add new edges (factors) to the factor graph
        
        This is the core method for building the factor graph by adding constraints
        between frames. Each factor represents a visual constraint between two frames.
        
        Args:
            ii: Source frame indices
            jj: Target frame indices  
            remove: Whether to remove old factors if max_factors limit is exceeded
        """
        # Convert to tensors if needed
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # Remove duplicate edges to avoid redundant factors
        ii, jj = self.__filter_repeated_edges(ii, jj)
        if ii.shape[0] == 0:
            return

        # Enforce maximum factor limit by removing oldest factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            # Remove oldest factors first (FIFO strategy)
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        # first128_cnet_feature (cnet) for source frames
        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # Build correlation volumes for visual feature matching
        if self.corr_impl == "volume": # by default, it is volume 
            c = (ii == jj).long() # yuhang: c should always be 0 vector, this looks like a bug
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)  # [1, n_edge, 128, ht (45), wd (80)]
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)  # [1, n_edge, 128, ht (45), wd (80)]

            corr = CorrBlock(fmap1, fmap2)  # correlation block for feature matching

            # [batch, num, 4*radius^2, ht, wd] = [batch, n_edge, 196, ht, wd]
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            # second128_cnet_feature for source frames
            inp = self.video.inps[ii].to(self.device).unsqueeze(0) # [1, n_edge, 128, ht (45), wd (80)]
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        # Compute reprojection targets and initialize weights
        with torch.amp.autocast('cuda', enabled=False):
            target, _ = self.video.reproject(ii, jj)  # project points in ii to jj frame, return the projected pixel locations
            weight = torch.zeros_like(target)  # initialize weights to zero

        # Add new factors to the graph
        self.ii = torch.cat([self.ii, ii], 0)  # source frame indices
        self.jj = torch.cat([self.jj, jj], 0)  # target frame indices
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)  # initialize age to 0

        # first128_cnet_feature
        self.net = net if self.net is None else torch.cat([self.net, net], 1)
        self.target = torch.cat([self.target, target], 1)  # target pixel locations
        self.weight = torch.cat([self.weight, weight], 1)  # confidence weights

    @torch.amp.autocast('cuda',enabled=True)
    def rm_factors(self, mask, store=False):
        """
        Remove factors (edges) from the factor graph
        
        Args:
            mask: Boolean mask indicating which factors to remove
            store: If True, store removed factors as inactive for potential reuse
        """
        # Store removed factors as inactive if requested
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        # Remove factors from active graph
        self.ii = self.ii[~mask]  # source frame indices
        self.jj = self.jj[~mask]  # target frame indices
        self.age = self.age[~mask]  # factor ages
        
        # Remove corresponding correlation data
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        # Remove network features and input data
        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        # Remove reprojection targets and weights
        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.amp.autocast('cuda',enabled=True)
    def rm_keyframe(self, ix):
        """
        Remove a keyframe from the video and update all factor graph indices
        
        This method removes a keyframe by shifting all subsequent frames up by one index
        and updating all factor graph references accordingly.
        
        Args:
            ix: Index of the keyframe to remove
        """
        # Update video data by shifting all frames after ix up by one
        with self.video.get_lock():
            # Basic frame data
            self.video.timestamp[ix] = self.video.timestamp[ix+1]
            self.video.images[ix] = self.video.images[ix+1]
            self.video.dirty[ix] = self.video.dirty[ix+1]
            self.video.npc_dirty[ix] = self.video.npc_dirty[ix+1]
            
            # Pose and depth data
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_up[ix] = self.video.disps_up[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]
            self.video.depth_scale[ix] = self.video.depth_scale[ix+1]
            self.video.depth_shift[ix] = self.video.depth_shift[ix+1]
            
            # Monocular depth data
            self.video.mono_disps[ix] = self.video.mono_disps[ix+1]
            self.video.mono_disps_up[ix] = self.video.mono_disps_up[ix+1]
            self.video.mono_disps_mask_up[ix] = self.video.mono_disps_mask_up[ix+1]
            self.video.valid_depth_mask[ix] = self.video.valid_depth_mask[ix+1]
            self.video.valid_depth_mask_small[ix] = self.video.valid_depth_mask_small[ix+1]

            # Feature data
            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

            # Uncertainty-aware features (if enabled)
            if self.video.uncertainty_aware:
                self.video.dino_feats[ix] = self.video.dino_feats[ix+1]
                self.video.uncertainties_inv[ix] = self.video.uncertainties_inv[ix+1]

        # Update inactive factors: remove those involving the deleted frame
        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1  # decrement indices >= ix
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            # Remove factors involving the deleted frame
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        # Update active factors: remove those involving the deleted frame
        m = (self.ii == ix) | (self.jj == ix)
        self.ii[self.ii >= ix] -= 1  # decrement indices >= ix
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)  # remove factors involving deleted frame


    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False,
               EP=1e-7, motion_only=False):
        """
        Run the main update operator on the factor graph
        
        This is the core optimization method that:
        1. Computes motion features from reprojection
        2. Extracts correlation features
        3. Runs the update operator to get pose/depth updates
        4. Performs bundle adjustment optimization
        
        Args:
            t0: Start frame for optimization (default: min frame + 1)
            t1: End frame for optimization (default: None for all frames)
            itrs: Number of bundle adjustment iterations
            use_inactive: Whether to include inactive factors in optimization
            EP: Epsilon for numerical stability
            motion_only: Whether to optimize only motion (not structure)
        """
        # Compute motion features from reprojection residuals
        with torch.amp.autocast('cuda', enabled=False):
            # project ii to jj, return the projected pixel locations [1, n_edges, ht, wd, 2]
            coords1, mask = self.video.reproject(self.ii, self.jj) 

            # coords1: projection of current frame to target frame during update
            # self.target: projection when adding new edges
            # self.coords0: raw pixel location map

            # so coords1 - self.coords0 should be how much each pixel moves from ii
            # self.target - coords1 should be how much the projection moves from the the raw projection
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1) # [1, n_edges, ht, wd, 4]
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)  # clamp for stability, [1, n_edges, 4, ht, wd]
        
        # correlation feature for reprojection
        corr = self.corr(coords1) # [1, n_edges, 196, ht(45), wd(80)]

        # get pose/depth updates
        # def forward(self, first128_cnet_feature, second128_cnet_feature, corr...
        # class UpdateModule(nn.Module)
        # update self.net using gru, corr, motn
        # delta is optical flow, [1, n_edges, ht, wd, 2]
        # weight is confidence weights, [1, n_edges, ht, wd, 2]
        # damping is Depth update magnitude for each pixel, [n_unique_frames of ii!, 45, 80]
        # upmask: Upsampling mask for depth refinement, [1, n_unique_frames_in_ii, 576, 45, 80]
        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        # Set default start frame if not provided
        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.amp.autocast('cuda',enabled=False):
            # Update targets and weights based on update operator output
            self.target = coords1 + delta.to(dtype=torch.float)  # new target locations
            self.weight = weight.to(dtype=torch.float)  # updated confidence weights

            # Update damping factors for optimization
            self.damping[torch.unique(self.ii)] = damping

            # Optionally include inactive factors in optimization
            if use_inactive:
                # Include recent inactive factors (within 3 frames of t0)
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
            else:
                # Use only active factors
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            # Prepare damping for bundle adjustment
            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            # Perform bundle adjustment optimization, optimize pose and depth
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                iters=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            # Upsample depth maps for updated frames
            self.video.upsample(torch.unique(self.ii), upmask)

        # Increment age of all factors
        self.age += 1


    @torch.amp.autocast('cuda',enabled=False)
    @torch.no_grad()
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8, enable_wq=True):
        """
        Memory-efficient update operator for large factor graphs
        
        This method processes factors in batches to reduce memory usage when dealing
        with large numbers of factors. It uses an alternative correlation implementation
        and processes factors in chunks.
        
        Args:
            t0: Start frame for optimization
            t1: End frame for optimization  
            itrs: Number of bundle adjustment iterations
            use_inactive: Whether to include inactive factors
            EP: Epsilon for numerical stability
            steps: Number of update steps
            enable_wq: Whether to enable weighted quantization (unused)
        """
        # Use alternative correlation implementation for memory efficiency
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        # Perform multiple update steps
        for step in range(steps):
            # Compute motion features
            with torch.amp.autocast('cuda', enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
            
            # Process factors in batches to reduce memory usage
            s = 8  # batch size
            for i in range(0, self.jj.max()+1, s):
                # Select factors in current batch
                v = (self.ii >= i) & (self.ii < i + s)
                if v.sum() < 1:
                    continue
                iis = self.ii[v]  # source frames in batch
                jjs = self.jj[v]  # target frames in batch

                ht, wd = self.coords0.shape[0:2]
                # Compute correlation for current batch
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.amp.autocast('cuda', enabled=True):
                    # Run update operator on current batch
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)
                    self.video.upsample(torch.unique(iis), upmask)

                # Update factors in current batch
                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            # Prepare for bundle adjustment
            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target
            weight = self.weight

            # Perform dense bundle adjustment            
            self.video.ba(target, weight, damping, self.ii, self.jj, t0, t1, 
                iters=itrs, lm=1e-5, ep=1e-2, motion_only=False)


    def add_neighborhood_factors(self, t0, t1, r=3):
        """
        Add factors between neighboring frames within a temporal radius
        
        This creates a dense connectivity pattern where each frame is connected
        to all frames within radius r. This is useful for maintaining local
        consistency in the factor graph.
        
        Args:
            t0: Start frame index
            t1: End frame index
            r: Temporal radius for neighborhood connections
        """
        # Create all possible frame pairs in the range [t0, t1)
        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1),indexing="ij")
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device) # 1d
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device) # 1d

        # Keep only pairs within temporal radius (excluding self-connections)
        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """
        Add factors based on visual similarity and temporal proximity
        
        This method intelligently selects frame pairs to connect based on:
        1. Visual similarity (using distance metric)
        2. Temporal proximity constraints
        3. Non-maximum suppression to avoid redundant connections
        
        Args:
            t0: Start frame for source frames
            t1: Start frame for target frames  
            rad: Temporal radius for local connections
            nms: Non-maximum suppression radius
            beta: Weight for distance calculation
            thresh: Distance threshold for factor selection
            remove: Whether to remove old factors when adding new ones
        """

        # video is DepthVideo object
        t = self.video.counter.value
        ix = torch.arange(t0, t)  # source frame indices
        jx = torch.arange(t1, t)  # target frame indices

        # Create all possible frame pairs
        ii, jj = torch.meshgrid(ix, jx,indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        # Compute visual distance between frame pairs
        # 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33...
        d = self.video.distance(ii, jj, beta=beta)
        
        # set diantance as infinite for close and far away frames
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        # ============================================================================
        # STEP 1: PRE-SUPPRESS BASED ON EXISTING FACTORS
        # ============================================================================
        # Goal: Prevent adding new factors that are too close to factors already in the graph
        # This is a "pre-filter" to avoid redundant connections with existing factors
        # 
        # WHY TWO NMS STEPS?
        # - Step 1: Suppress based on factors already in the graph (from previous calls)
        # - Step 5: Suppress based on factors selected in THIS call (within the same iteration)
        # This ensures we don't create redundant factors either with existing ones OR with
        # other factors we're adding in the same batch
        # 
        # Example scenario:
        # - We have existing factors: (2,5), (3,6), (1,4) [bad], (4,7) [inactive]
        # - We want to add new factors between frames t0=2 to t1=8
        # - nms=2 means suppress factors within 2-frame radius of existing ones
        
        # Step 1: Collect ALL existing frame pairs (active + bad + inactive)
        # This includes every factor that has ever been considered
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)  # source frames
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)  # target frames
        
        # Example: ii1 = [2, 3, 1, 4], jj1 = [5, 6, 4, 7]
        # These represent existing factors: (2,5), (3,6), (1,4), (4,7)
        
        # Step 2: For each existing factor, suppress nearby candidates
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            # i, j = existing factor (ii1[i], jj1[j])
            # Example: i=2, j=5 (existing factor from frame 2 to frame 5)
            
            # Step 3: Check all positions within nms radius around (i, j)
            for di in range(-nms, nms+1):  # di ∈ [-2, -1, 0, 1, 2] if nms=2
                for dj in range(-nms, nms+1):  # dj ∈ [-2, -1, 0, 1, 2] if nms=2
                    # di, dj = offset from existing factor (i, j)
                    # We're checking position (i+di, j+dj)
                    
                    # Step 4: Calculate suppression radius based on temporal distance
                    # The closer frames are in time, the smaller the suppression radius
                    temporal_distance = abs(i - j)  # How far apart are source and target?
                    suppression_radius = max(min(temporal_distance - 2, nms), 0)
                    
                    # Example: For factor (2,5):
                    # - temporal_distance = |2-5| = 3
                    # - suppression_radius = max(min(3-2, 2), 0) = max(min(1, 2), 0) = 1
                    # - So we only suppress positions where |di| + |dj| <= 1
                    
                    if abs(di) + abs(dj) <= suppression_radius:
                        # This position (i+di, j+dj) is too close to existing factor (i, j)
                        i1 = i + di  # candidate source frame
                        j1 = j + dj  # candidate target frame
                        
                        # Example: For existing factor (2,5) with suppression_radius=1:
                        # - (2,5) gets suppressed (di=0, dj=0, |0|+|0|=0 <= 1) ✓
                        # - (1,5) gets suppressed (di=-1, dj=0, |-1|+|0|=1 <= 1) ✓
                        # - (3,5) gets suppressed (di=1, dj=0, |1|+|0|=1 <= 1) ✓
                        # - (2,4) gets suppressed (di=0, dj=-1, |0|+|-1|=1 <= 1) ✓
                        # - (2,6) gets suppressed (di=0, dj=1, |0|+|1|=1 <= 1) ✓
                        # - (1,4) gets suppressed (di=-1, dj=-1, |-1|+|-1|=2 > 1) ✗
                        # - (3,6) gets suppressed (di=1, dj=1, |1|+|1|=2 > 1) ✗
                        
                        # Step 5: Check if candidate is within our target range
                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            # Convert 2D coordinates (i1, j1) to 1D index in distance array
                            # Distance array is flattened: d[source_index * num_targets + target_index]
                            array_index = (i1 - t0) * (t - t1) + (j1 - t1)
                            d[array_index] = np.inf  # Mark as "infinite distance" = don't select
                            
                            # Example: If t0=2, t1=2, t=8:
                            # - For candidate (2,5): array_index = (2-2)*6 + (5-2) = 0*6 + 3 = 3
                            # - For candidate (3,5): array_index = (3-2)*6 + (5-2) = 1*6 + 3 = 9
                            # - Setting d[3] = inf and d[9] = inf means these won't be selected

        # ============================================================================
        # STEP 2: ADD LOCAL NEIGHBORHOOD CONNECTIONS (within temporal radius)
        # ============================================================================
        # Goal: Add connections between temporally close frames (within 'rad' frames)
        # These are "obvious" connections that should always be present
        
        es = []  # List to store all selected factor pairs (source, target)
        
        # Add local connections: connect each frame to nearby frames within 'rad' distance
        for i in range(t0, t):  # For each source frame from t0 to t
            for j in range(max(i-rad-1, 0), i):  # For each target frame within 'rad' of source
                # Example: If i=5, rad=2, then j ∈ [max(5-2-1, 0), 5) = [2, 5) = [2, 3, 4]
                # This creates connections: (5,2), (5,3), (5,4)
                
                es.append((i, j))  # Forward connection: i → j
                es.append((j, i))  # Backward connection: j → i (bidirectional)
                
                # Mark these local connections as "infinite distance" so they won't be
                # selected again in the distance-based selection below
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf
                
                # Example: For connection (5,2) with t0=2, t1=2, t=8:
                # array_index = (5-2)*6 + (2-2) = 3*6 + 0 = 18
                # Setting d[18] = inf means this won't be selected again

        # ============================================================================
        # STEP 3: SELECT ADDITIONAL FACTORS BASED ON VISUAL DISTANCE
        # ============================================================================
        # Goal: Select the best remaining factors based on visual similarity
        # (after excluding local connections and NMS-suppressed ones)
        
        # Sort all candidate factors by their visual distance (ascending = best first)
        ix = torch.argsort(d)  # Indices sorted by distance: closest factors first
        
        # Example: If d = [inf, 5.2, inf, 3.1, 7.8, inf, 2.9, ...]
        # Then ix = [6, 3, 1, 4, ...] (indices of d sorted by value)
        
        for k in ix:  # Process each candidate factor in order of quality
            # k = index in the flattened distance array
            # ii[k], jj[k] = the actual frame pair (source, target)
            
            # Skip if visual distance is too large (not similar enough)
            if d[k].item() > thresh:
                continue  # This factor is too dissimilar, skip it
                
            # Example: If thresh=16.0 and d[k]=20.5, skip this factor
            
            # Respect maximum factor limit to keep computation manageable
            if len(es) > self.max_factors:
                break  # We've reached the limit, stop adding more factors
                
            # Get the actual frame indices for this factor
            i = ii[k]  # source frame index
            j = jj[k]  # target frame index
            
            # Example: If k=6 corresponds to frames (3,7), then i=3, j=7
            
            # ========================================================================
            # STEP 4: ADD THE SELECTED FACTOR (with bidirectional connections)
            # ========================================================================
            # Add both forward and backward connections for this factor
            es.append((i, j))  # Forward: i → j
            es.append((j, i))  # Backward: j → i
            
            # Example: For factor (3,7), we add both (3,7) and (7,3) to es

            # ========================================================================
            # STEP 5: POST-SUPPRESS BASED ON NEWLY SELECTED FACTOR
            # ========================================================================
            # Immediately suppress nearby factors to avoid redundancy
            # This prevents selecting multiple similar factors in the same iteration
            # (Different from Step 1: this suppresses based on factors selected in THIS call)
            
            for di in range(-nms, nms+1):  # Check all positions within nms radius
                for dj in range(-nms, nms+1):
                    # Calculate suppression radius based on temporal distance
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di  # nearby source frame
                        j1 = j + dj  # nearby target frame
                        
                        # Suppress this nearby position if it's within our target range
                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf
                            
                            # Example: If we just selected (3,7) and nms=2:
                            # - Suppress (3,7) itself (di=0, dj=0)
                            # - Suppress (2,7), (4,7) (1 frame away in source)
                            # - Suppress (3,6), (3,8) (1 frame away in target)
                            # - Suppress (2,6), (2,8), (4,6), (4,8) (diagonal neighbors)

        # ============================================================================
        # STEP 6: ADD ALL SELECTED FACTORS TO THE FACTOR GRAPH
        # ============================================================================
        # Convert the list of factor pairs to tensors and add them to the graph
        
        # Convert es (list of tuples) to separate source and target tensors
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        # ii = tensor of source frame indices
        # jj = tensor of target frame indices
        
        # Example: If es = [(2,5), (5,2), (3,6), (6,3), (4,7), (7,4)]
        # Then ii = [2, 5, 3, 6, 4, 7] and jj = [5, 2, 6, 3, 7, 4]
        
        # Add these factors to the actual factor graph
        self.add_factors(ii, jj, remove)
        # 'remove' parameter determines whether to remove old factors when adding new ones


    def add_backend_proximity_factors(self, t_start, t_end, nms, radius, thresh, max_factors, beta, t_start_loop=None, loop=False):
        """
        Advanced proximity factor addition with loop closure support
        
        This method is designed for backend processing where more sophisticated
        factor selection is needed, including support for loop closure detection.
        
        Args:
            t_start: Start frame for target frames
            t_end: End frame for both source and target
            nms: Non-maximum suppression radius
            radius: Temporal radius for local connections
            thresh: Distance threshold for factor selection
            max_factors: Maximum number of factors to add
            beta: Weight for distance calculation
            t_start_loop: Start frame for loop closure search
            loop: Whether to enable loop closure detection
            
        Returns:
            Number of edges added to the factor graph
        """
        if t_start_loop is None or not loop:
            t_start_loop = t_start
        assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'

        # Define frame ranges
        ilen = (t_end - t_start_loop)  # source frame range length
        jlen = (t_end - t_start)       # target frame range length
        ix = torch.arange(t_start_loop, t_end)  # source frames
        jx = torch.arange(t_start, t_end)      # target frames

        # Create all possible frame pairs
        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        # Compute visual distances
        d = self.video.distance(ii, jj, beta=beta)
        rawd = deepcopy(d).reshape(ilen, jlen)  # keep original distances for loop detection
        
        # Set infinite distance for frames too close in time
        d[ii - radius < jj] = np.inf
        d[d > thresh] = np.inf
        d = d.reshape(ilen, jlen)

        es = []
        # Build local neighborhood connections within temporal radius
        for i in range(t_start_loop, t_end):
            for j in range(max(i-radius-1, 0), i):  # j in [i-radius, i-1]
                es.append((i, j))  # forward connection
                es.append((j, i))  # backward connection
                di, dj = i-t_start_loop, j-t_start
                d[di, dj] = np.inf  # mark as used

        # Sort by distance and select factors
        vals, ix = torch.sort(d.reshape(-1), descending=False)
        ix = ix[vals<=thresh]  # only consider factors within threshold
        ix = ix.tolist()

        loop_edges = 0
        n_neighboring = 1  # neighborhood size for loop closure

        # Add factors based on distance
        for k in ix:
            di, dj = k // jlen, k % jlen
            if d[di,dj].item() > thresh:
                continue

            if len(es) > max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            if loop:
                # Loop closure: add factors in neighborhood of selected pair
                sub_es = []
                num_loop = 0
                for si in range(max(i-n_neighboring, t_start_loop), min(i+n_neighboring+1, t_end)):
                    for sj in range(max(j-n_neighboring, t_start), min(j+n_neighboring+1, t_end)):
                        if rawd[(si-t_start_loop), (sj-t_start)] <= thresh:
                            num_loop += 1
                            # Only add if frames are sufficiently far apart (loop closure)
                            if si != sj and si-sj > 20:
                                sub_es += [(si, sj)]
                es += sub_es
                loop_edges += len(sub_es)
            else:
                # Regular bidirectional connection
                es += [(i, j), ]
                es += [(j, i), ]

            # Apply NMS around selected factor
            d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        # Check if we have enough factors
        if len(es) < 3 or (loop and loop_edges==0):
            return 0

        # Add selected factors to the graph
        ii, jj = torch.tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove=True)
        edge_num = len(self.ii)

        return edge_num