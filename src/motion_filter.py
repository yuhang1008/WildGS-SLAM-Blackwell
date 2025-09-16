import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor

class MotionFilter:
    """
    Motion-based keyframe selection and feature extraction for SLAM.
    
    This class is the first stage of the SLAM pipeline that:
    1. Extracts visual features from input images using DroidNet components
    2. Estimates motion between consecutive frames
    3. Decides whether to add a new keyframe based on motion threshold
    4. Predicts metric depth and DINO features for uncertainty estimation
    
    The class uses three components from DroidNet:
    - fnet: Feature network for visual correspondence matching
    - cnet: Context network for iterative pose/depth refinement  
    - update: Update module for pose and depth estimation
    
    Mainly inherited from DROID-SLAM architecture.
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        """
        Initialize the motion filter with DroidNet components.
        
        Args:
            net: DroidNet instance containing fnet, cnet, and update modules
            video: DepthVideo instance for storing SLAM state
            cfg: Configuration dictionary
            thresh: Motion threshold for keyframe selection (default: 2.5)
            device: Computing device (default: "cuda:0")
        """
        self.cfg = cfg
        
        # Extract DroidNet components for feature extraction and updates
        # These are the core neural networks from DROID-SLAM
        # net is instance of DroidNet

        # self.fnet = BasicEncoder(out_dim=128, norm_fn='instance') #b, n, 3, h, w ->  b, n, 128, h1/8, w1/8
        # self.cnet = BasicEncoder(out_dim=256, norm_fn='none') #b, n, 3, h, w ->  b, n, 256, h1/8, w1/8
        self.cnet = net.cnet      # Context network: encodes contextual information for updates
        self.fnet = net.fnet      # Feature network: extracts features for correlation matching
        self.update = net.update  # Update module: performs iterative pose/depth refinement

        # SLAM state and parameters
        self.video = video        # Shared memory buffer for poses, depths, and features
        self.thresh = thresh      # Motion threshold for keyframe selection
        self.device = device      # Computing device
        self.count = 0            # Counter for consecutive dropped frames
        
        # Context features for temporal consistency
        self.prev_kf_first128_cnet_feature = None  # Hidden state features from previous keyframe
        self.prev_kf_second128_cnet_feature = None  # Input features from previous keyframe
        self.pre_kf_fnet_feature = None  # Visual features from previous keyframe for correlation

        # Image normalization parameters (ImageNet standard)
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        # Uncertainty-aware processing setup
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        
        # Initialize metric depth estimator (Metric3D or Depth Anything V2)
        self.metric_depth_estimator = get_metric_depth_estimator(cfg)
        
        # Initialize DINO feature extractor for uncertainty estimation
        if cfg['mapping']["uncertainty_params"]['activate']:
            # If mapping needs dino features, we still need feature extractor
            self.feat_extractor = get_feature_extractor(cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """
        Extract contextual features using the context network (cnet).
        
        The cnet produces 256-dimensional features that are split into:
        - first128_cnet_feature (128-dim): Hidden state for temporal consistency (tanh activated)
        - second128_cnet_feature (128-dim): Input features for observations (ReLU activated)
        
        These features are used by the update module for iterative pose/depth refinement.
        
        Args:
            image: Normalized input image tensor
            
        Returns:
            tuple: (first128_cnet_feature, second128_cnet_feature) - context features for the update module
        """

        # cnet: b, n, 3, h, w ->  b, n, 256, h1/8, w1/8
        first128_cnet_feature, second128_cnet_feature = self.cnet(image).split([128,128], dim=2)
        return first128_cnet_feature.tanh().squeeze(0), second128_cnet_feature.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """
        Extract visual features using the feature network (fnet).
        
        The fnet produces 128-dimensional dense features that are used for:
        - Building correlation volumes between frames
        - Visual correspondence matching
        - Optical flow estimation
        
        Args:
            image: Normalized input image tensor
            
        Returns:
            torch.Tensor: 128-dimensional feature maps for correlation matching
        """
        return self.fnet(image).squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """
        Main tracking operation that processes each incoming frame.
        
        This function implements the core motion filtering pipeline:
        1. Extract visual features using fnet
        2. Estimate motion between current and previous frame
        3. Decide whether to add current frame as keyframe
        4. Predict metric depth and DINO features (uncertainty) if keyframe is selected
        
        The decision to add a keyframe is based on:
        - Motion magnitude threshold (self.thresh)
        - Forced keyframe interval (if configured)
        - First frame is always added
        
        Args:
            tstamp: Timestamp of the current frame
            image: Input RGB image tensor
            intrinsics: Camera intrinsic parameters
            
        Returns:
            bool: Whether the current frame was added as a keyframe
        """
        # Initialize identity pose for first frame
        Id = lietorch.SE3.Identity(1,).data.squeeze()
        
        # Calculate downsampled image dimensions
        ht = image.shape[-2] // self.video.down_scale # 360 / 8 = 45
        wd = image.shape[-1] // self.video.down_scale # 640 / 8 = 80

        # image shape: [1, 3, 360, 640]
        # inputs shape: [1, 1, 3, 360, 640]
        inputs = image[None, ...].to(self.device)
        # Normalize input image using ImageNet statistics
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # Step 1: Extract visual features using the feature network (fnet)
        # These features will be used for correlation matching
        fnet_feature = self.__feature_encoder(inputs) # 1, 1, 3, h, w ->  1, 1, 128, h1/8, w1/8

        force_to_add_keyframe = False

        ### Handle first frame - always add to initialize the system ###
        if self.video.counter.value == 0:
            # Extract context features using the context network (cnet)
            first128_cnet_feature, second128_cnet_feature = self.__context_encoder(inputs[:,[0]])
            self.prev_kf_first128_cnet_feature = first128_cnet_feature
            self.prev_kf_second128_cnet_feature = second128_cnet_feature
            self.pre_kf_fnet_feature = fnet_feature
            
            # Predict metric depth using Metric3D or Depth Anything V2
            # by default is metric3d_vit_large
            # image: [1, 3, 360, 640]
            # mono_depth: [360, 640]
            mono_depth = predict_metric_depth(self.metric_depth_estimator, tstamp, image, self.cfg, self.device)
            
            # Extract DINO features for uncertainty estimation. default: true
            if self.uncertainty_aware:
                # dino_features [25(h/14), 45(w/14), 384]
                dino_features = predict_img_features(self.feat_extractor, tstamp, image, self.cfg, self.device)
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    # If mapping needs dino features, we predict here and store the value in local disk
                    _ = predict_img_features(self.feat_extractor, tstamp, image, self.cfg, self.device)
            
            # Add first frame to the video buffer with all extracted features
            # self.video is an instance of class DepthVideo
            self.video.append(tstamp, 
                              image[0],  # [3, 360, 640]
                              Id, # [7], identity pose for the first frame
                              1.0,
                              mono_depth, # [360, 640], from Metric3D or Depth Anything V2, use raw image as input
                              intrinsics / float(self.video.down_scale), 
                              fnet_feature, # [128, (45)h/8, (80)w/8] from fnet, use normalized image as input
                              first128_cnet_feature[0,0], # [128, (45)h/8, (80)w/8] from cnet, use normalized image as input
                              second128_cnet_feature[0,0], # [128, (45)h/8, (80)w/8] from cnet, use normalized image as input
                              dino_features) # [25(h/14), 45(w/14), 384] from dino, use normalized raw image as input
                            
        ### Handle subsequent frames - only add if there's enough motion ###
        else:                
            # Step 2: Build correlation volume between previous and current frame
            # This measures visual correspondence between frames
            
            # coords0[row, col] = [col, row]
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            
            # pass fnet feature of last and current frame, generate image pyramid correlation features
            # saved in corr.corr_pyramid, shape:
            #[
            # [1, 45, 80, 45, 80]
            # [1, 45, 80, 22, 40]
            # [1, 45, 80, 11, 20]
            # [1, 45, 80, 5, 10]
            #]

            # [1, 1, 196, 45, 80] 196 = (7*7)*4, features from 4 pyramid levels and concat
            corr = CorrBlock(self.pre_kf_fnet_feature[None,[0]], fnet_feature[None,[0]])(coords0)

            # Step 3: Estimate pixel displacement using the update module
            # prev_first128_cnet_feature + prev_second128_cnet_feature + corr → delta: pixel motion estimate
            # droid_net.py, def forward(self, first128_cnet_feature, second128_cnet_feature, corr, flow=None, ii=None, jj=None)
            _, delta, weight = self.update(self.prev_kf_first128_cnet_feature[None], self.prev_kf_second128_cnet_feature[None], corr)
            # delta: [b, n, 45, 80, 2], weight: [b, n, 45, 80, 2]

            # Check if we should force a keyframe based on frame interval
            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']

            # Step 4: Decide whether to add current frame as keyframe
            # Decision is based on motion magnitude or forced keyframe interval
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                # Motion is significant enough - add as keyframe
                self.count = 0
                
                # Extract context features for the new keyframe (cnet)
                first128_cnet_feature, second128_cnet_feature = self.__context_encoder(inputs[:,[0]])
                self.prev_kf_first128_cnet_feature = first128_cnet_feature
                self.prev_kf_second128_cnet_feature = second128_cnet_feature
                self.pre_kf_fnet_feature = fnet_feature
                
                # Predict metric depth for the new keyframe
                # [360, 640]
                mono_depth = predict_metric_depth(self.metric_depth_estimator, tstamp, image, self.cfg, self.device)
                
                # Extract DINO features for uncertainty estimation (if enabled)
                if self.uncertainty_aware:
                    #[25(h/14), 45(w/14), 384]
                    dino_features = predict_img_features(self.feat_extractor, tstamp, image, self.cfg, self.device)
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        # if mapping needs dino features, we predict here and store the value in local disk
                        _ = predict_img_features(self.feat_extractor, tstamp, image, self.cfg, self.device)
                
                # Add new keyframe to the video buffer
                self.video.append(
                    tstamp, 
                    image[0],  # [3, 360, 640]
                    None, # pose
                    None,
                    mono_depth, # [360, 640], from Metric3D or Depth Anything V2, use raw image as input
                    intrinsics / float(self.video.down_scale), 
                    fnet_feature, # [128, (45)h/8, (80)w/8] from fnet, use normalized image as input
                    first128_cnet_feature[0], # [128, (45)h/8, (80)w/8] from cnet, use normalized image as input
                    second128_cnet_feature[0], # [128, (45)h/8, (80)w/8] from cnet, use normalized image as input
                    dino_features) # [25(h/14), 45(w/14), 384] from dino, use normalized raw image as input
            else:
                # Motion is too small - skip this frame
                self.count += 1

        return force_to_add_keyframe

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        """
        Extract DINO features for uncertainty estimation.
        
        This method is used to extract DINO features from images, typically
        for full-resolution processing or when features are needed separately
        from the main tracking pipeline.
        
        Args:
            tstamp: Timestamp of the image
            image: Input RGB image tensor
            suffix: Optional suffix for saving features (e.g., 'full' for full resolution)
            
        Returns:
            torch.Tensor: DINO features for uncertainty estimation
        """
        dino_features = predict_img_features(self.feat_extractor, tstamp, image, self.cfg, self.device, suffix=suffix)
        return dino_features
