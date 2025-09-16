from src.motion_filter import MotionFilter
from src.frontend import Frontend 
from src.backend import Backend
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net # DroidNet
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.output = slam.save_dir

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)
        self.online_ba = Backend(self.net,self.video, self.cfg)
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']

        self.printer:Printer = slam.printer

    def run(self, stream:BaseDataset):
        """
        Main tracking loop that processes the input image stream.
        
        This function implements the core tracking pipeline:
        1. Motion filtering: Check if there's enough motion to add a new keyframe
        2. Frontend processing: Local bundle adjustment and keyframe selection
        3. Backend optimization: Periodic global bundle adjustment
        4. Inter-process communication: Send pose/depth estimates to mapper
        
        Args:
            stream (BaseDataset): Input image stream containing timestamps and images
        """
        # Initialize tracking state variables
        prev_kf_idx = 0      # Previous keyframe index
        curr_kf_idx = 0      # Current keyframe index  
        prev_ba_idx = 0      # Last keyframe where bundle adjustment was performed

        # Get camera intrinsic parameters
        intrinsic = stream.get_intrinsic()
        
        # Process each frame in the input stream
        for i in range(len(stream)):
            timestamp, image, _, _ = stream[i]
            
            with torch.no_grad():
                # Store the keyframe count before processing this frame
                starting_count = self.video.counter.value
                
                # Step 1: Motion filtering - determine if frame should be added as keyframe
                # This checks if there's sufficient motion between current frame and last keyframe
                # if to make kf, save features to video buffer
                force_to_add_keyframe = self.motion_filter.track(timestamp, image, intrinsic)

                # Step 2: Frontend processing - local bundle adjustment
                # This estimates camera pose and depth, and may remove redundant keyframes
                # see def __update(self, force_to_add_keyframe) in frontend.py
                self.frontend(force_to_add_keyframe)

                # Step 3: Handle full-resolution features if enabled and new keyframe was added
                if (starting_count < self.video.counter.value) and self.cfg['mapping']['full_resolution']:
                    if self.motion_filter.uncertainty_aware:
                        # Extract full-resolution image features for uncertainty estimation
                        img_full = stream.get_color_full_resol(i)
                        self.motion_filter.get_img_feature(timestamp, img_full, suffix='full')
            
            # Update current keyframe index
            curr_kf_idx = self.video.counter.value - 1
            
            # Step 4: Handle new keyframes and communicate with mapper
            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                if self.video.counter.value == self.frontend.warmup:
                    # Special case: SLAM system initialization just completed
                    self.pipe.send({
                        "is_keyframe": True, 
                        "video_idx": curr_kf_idx,
                        "timestamp": timestamp,
                        "just_initialized": True, 
                        "end": False
                    })
                    self.pipe.recv()  # Wait for mapper acknowledgment
                    self.frontend.initialize_second_stage()  # Start second stage initialization
                else:
                    # Regular keyframe processing
                    if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                        # Step 5: Periodic global bundle adjustment
                        # Run global BA every ba_freq keyframes to maintain global consistency
                        self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}", FontColor.TRACKER)
                        self.online_ba.dense_ba(2)
                        prev_ba_idx = curr_kf_idx
                    
                    # Step 6: Send pose and depth estimates to mapper process
                    # This triggers the mapping process to update the 3D map
                    self.pipe.send({
                        "is_keyframe": True,
                        "video_idx": curr_kf_idx,
                        "timestamp": timestamp,
                        "just_initialized": False, 
                        "end": False
                    })
                    self.pipe.recv()  # Wait for mapper to finish processing

            # Update state for next iteration
            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()  # Update progress bar

        # Step 7: Signal end of tracking to mapper process
        self.pipe.send({
            "is_keyframe": True, 
            "video_idx": None,
            "timestamp": None, 
            "just_initialized": False, 
            "pose": None,
            "depth": None,
            "end": True
        })


                