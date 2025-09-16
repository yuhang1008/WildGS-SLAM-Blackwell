import torch
from src.factor_graph import FactorGraph
from src.backend import Backend as LoopClosing

class Frontend:
    # mainly inherited from GO-SLAM
    def __init__(self, net, video, cfg):
        self.cfg = cfg
        self.video = video
        self.update_op = net.update
        
        # local optimization window
        self.t1 = 0

        # frontent variables
        self.is_initialized = False

        self.max_age = cfg['tracking']['max_age']
        self.iters1 = 4*2
        self.iters2 = 2*2

        self.warmup = cfg['tracking']['warmup']
        self.beta = cfg['tracking']['beta']
        self.frontend_nms = cfg['tracking']['frontend']['nms']
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh']
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.frontend_thresh = cfg['tracking']['frontend']['thresh']
        self.frontend_radius = cfg['tracking']['frontend']['radius']
        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors']

        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, cfg)

        self.graph = FactorGraph(
            video, net.update,
            device=cfg['device'],
            corr_impl='volume',
            max_factors=self.frontend_max_factors
        )

        ## This is to avoid too many consecutive candidate keyframes which:
        #  1. capture large moving objects (high optical flow)
        #  2. don't have much camera motion (will be removed from the candidate later on)
        ## If there are too many of this kind of keyframes, we will have 0 edge in the graph.
        #  Because when a frame is determined as potential keyframe, other edges will be updated as well
        #  even if this frame is removed at the end due to less camera motion. And we will remove the edges
        #  that have been updated more than cfg['tracking']['max_age']
        self.max_consecutive_drop_of_keyframes = (cfg['tracking']['max_age']/self.iters1)//3
        self.num_keyframes_dropped = 0

    def __update(self, force_to_add_keyframe):
        """
        Frontend update operation for SLAM pose and depth optimization.
        
        This function performs the core frontend processing after a new keyframe is added:
        1. Manages the factor graph (adds/removes edges between keyframes)
        2. Performs bundle adjustment to optimize poses and depths
        3. Decides whether to keep or drop the new keyframe
        4. Handles loop closing for long-term consistency
        5. Updates visualization and memory management
        
        Args:
            force_to_add_keyframe: Whether this keyframe was forced (e.g., by frame interval)
        """
        
        # ============================================================================
        # STEP 1: INCREMENT FRAME COUNTER AND CLEAN UP OLD FACTORS
        # ============================================================================
        self.t1 += 1  # Increment current frame index
        
        # Remove old factors from the factor graph to prevent memory growth
        # Factors older than max_age are moved to inactive storage
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # ============================================================================
        # STEP 2: ADD NEW FACTORS TO THE FACTOR GRAPH
        # ============================================================================
        # Add proximity factors between recent keyframes
        # This creates edges in the factor graph for bundle adjustment
        # - t1-5 to max(t1-frontend_window, 0): Connect recent keyframes
        # - rad: Radius for factor creation
        # - nms: Non-maximum suppression threshold
        # - thresh: Motion threshold for factor creation
        # - beta: Weight parameter for factor strength
        # - remove=True: Remove conflicting factors

        # add new factors and correspondig projection, features
        self.graph.add_proximity_factors(
            self.t1-5, 
            max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True)

        # ============================================================================
        # STEP 3: PERFORM INITIAL BUNDLE ADJUSTMENT
        # ============================================================================
        # Run bundle adjustment to optimize poses and depths using the factor graph
        # This refines the camera poses and depth estimates based on visual correspondences
        for itr in range(self.iters1): # self.iters1 = 4*2
            self.graph.update(None, None, use_inactive=True)

            # Optional: Filter high-error monocular depth estimates
            # This improves depth quality by removing unreliable depth predictions
            if not self.cfg['fast_mode']:
                if itr == 1 and self.video.metric_depth_reg and self.cfg['tracking']["uncertainty_params"]['activate']:
                    self.video.filter_high_err_mono_depth(self.t1-1,self.graph.ii,self.graph.jj)

        # ============================================================================
        # STEP 4: DECIDE WHETHER TO KEEP OR DROP THE NEW KEYFRAME
        # ============================================================================
        # Calculate motion distance between the last two keyframes
        # This measures how much the camera moved between keyframes
        d = self.video.distance([self.t1-2], [self.t1-1], beta=self.beta, bidirectional=True)
        
        # KEYFRAME DROPPING DECISION:
        # Drop the keyframe if:
        # 1. Motion is too small (d < keyframe_thresh) - not enough new information
        # 2. We haven't dropped too many consecutive keyframes (prevent over-dropping)
        # 3. This wasn't a forced keyframe (respect user's decision to force)
        if (d.item() < self.keyframe_thresh) & (self.num_keyframes_dropped < self.max_consecutive_drop_of_keyframes) & (not force_to_add_keyframe):
            # DROP KEYFRAME: Remove it from the factor graph and decrement counters
            self.graph.rm_keyframe(self.t1 - 1)         
            self.num_keyframes_dropped += 1
            with self.video.get_lock():
                self.video.counter.value -= 1  # Decrement total keyframe count
                self.t1 -= 1  # Decrement current frame index
        else:
            # KEEP KEYFRAME: Continue with additional processing
            cur_t = self.video.counter.value
            self.num_keyframes_dropped = 0  # Reset drop counter
            
            # ========================================================================
            # STEP 5: LOOP CLOSING AND ADDITIONAL BUNDLE ADJUSTMENT
            # ========================================================================
            if self.enable_loop and cur_t > self.frontend_window:
                # LOOP CLOSING: Detect and handle loop closures for long-term consistency
                # This is crucial for preventing drift in long sequences
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, 
                                                         motion_only=False, local_graph=self.graph,
                                                         enable_wq=True)
                
                # If no loop closures were found, run additional bundle adjustment
                if n_edge == 0:
                    for itr in range(self.iters2):
                        self.graph.update(t0=None, t1=None, use_inactive=True)
                self.last_loop_t = cur_t
            else:
                # NO LOOP CLOSING: Just run additional bundle adjustment
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True)

        # ============================================================================
        # STEP 6: INITIALIZE POSE AND DEPTH FOR NEXT ITERATION
        # ============================================================================
        # Set initial pose and depth estimates for the next frame
        # This provides a starting point for the next optimization
        self.video.poses[self.t1] = self.video.poses[self.t1-1]  # Copy pose from previous frame
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()  # Initialize depth with mean of previous

        # ============================================================================
        # STEP 7: UPDATE VISUALIZATION AND MEMORY MANAGEMENT
        # ============================================================================
        # Mark the affected region as dirty for visualization updates
        self.video.set_dirty(self.graph.ii.min(), self.t1)
        
        # Clear GPU cache to prevent memory accumulation
        torch.cuda.empty_cache()

    def __initialize(self):
        """ initialize the SLAM system, i.e. bootstrapping """

        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.set_dirty(0, self.t1)

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def initialize_second_stage(self):
        """ 2nd stage of initializing the SLAM system after we have reliable uncertainty mask from mapping """
        self.t1 = self.video.counter.value

        # update mask
        if self.cfg['tracking']["uncertainty_params"]['activate']:
            self.video.update_all_uncertainty_mask()

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # we don't want the edges from initialization start with very old age
        self.graph.age = torch.maximum(self.graph.age-8, torch.tensor(0).to(self.graph.age.device))

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.set_dirty(0, self.t1)

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self, force_to_add_keyframe):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup: # warmup is 12
            self.__initialize()
            self.video.update_valid_depth_mask()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            if self.cfg['tracking']["uncertainty_params"]['activate']:
                self.video.update_all_uncertainty_mask()
            self.__update(force_to_add_keyframe)
            self.video.update_valid_depth_mask()

