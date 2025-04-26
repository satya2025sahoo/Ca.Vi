import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
from collision_detector3 import EnhancedCollisionDetector
from instance_segmenter3 import InstanceSegmenter
import path_track

class EnhancedNavigationSystem:
    def __init__(self, video_path="video.mp4", output_dir="output", frames_dir="frames", depth_dir="depth_maps"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_dir = frames_dir
        self.depth_dir = depth_dir
        
        # Initialize models
        self.detector = EnhancedCollisionDetector()
        self.segmenter = InstanceSegmenter()
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # Progress tracking variables
        self.start_time = None
        self.processed_frames = 0
        self.total_frames = 0
        
        # Collision tracking
        self.collision_history = []
        self.warning_history = []
        self.collision_log_path = os.path.join(output_dir, "collision_log.csv")
        self.frame_status_log_path = os.path.join(output_dir, "frame_status.csv")  # New CSV for frame status
        self._init_collision_log()
        self._init_frame_status_log()

        # Distance thresholds
        self.warning_dist = 10.0
        self.stop_dist = 6.5
        self.roi_coverage_threshold = 0.1  # 10% of ROI area
        self.unknown_obstacle_threshold = 7.5  # Threshold for unknown obstacles
        self.large_size_threshold = 0.9  # 90% of ROI area
        self.min_cross_section_threshold = 0.4  # 40% cross-section overlap with ROI

    def _init_collision_log(self):
        """Initialize the collision log file"""
        if not os.path.exists(self.collision_log_path):
            with open(self.collision_log_path, 'w') as f:
                f.write("frame_idx,timestamp,status,object_class,distance,roi_coverage\n")

    def _init_frame_status_log(self):
        """Initialize the frame status log file"""
        if not os.path.exists(self.frame_status_log_path):
            with open(self.frame_status_log_path, 'w') as f:
                f.write("frame_idx,status\n")

    def _extract_frames(self, target_fps=6):
        """Extract frames from video at specified FPS"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / target_fps))
        
        frame_idx = 0
        saved_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(self.frames_dir, f"frame_{saved_count:05d}.jpg")
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, frame)
                saved_count += 1
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames at {target_fps} FPS")
        return saved_count

    def _get_depth_map(self, frame, frame_idx):
        """Get depth map for frame (from cache or compute new)"""
        npy_path = os.path.join(self.depth_dir, f"depth_{frame_idx:05d}.npy")
        if os.path.exists(npy_path):
            return np.load(npy_path)
        
        depth_map = self.detector.get_depth_map(frame)
        np.save(npy_path, depth_map)
        
        # Save visualization
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(npy_path.replace('.npy', '.jpg'), 
                   cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET))
        return depth_map

    def _display_progress(self):
        """Display progress"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        remaining_frames = self.total_frames - self.processed_frames
        eta = timedelta(seconds=int(remaining_frames / fps)) if fps > 0 else timedelta(0)
        progress = (self.processed_frames / self.total_frames) * 100
        
        progress_bar = f"[{'=' * int(progress/2):<50}] {progress:.1f}%"
        status_line = (f"Processed: {self.processed_frames}/{self.total_frames} | "
                      f"FPS: {fps:.1f} | "
                      f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                      f"ETA: {eta}")
        print(f"\r{progress_bar} {status_line}", end='', flush=True)

    def _log_collision(self, frame_idx, status, threats):
        """Log collision events to collision_log.csv and base status to frame_status.csv"""
        timestamp = time.time()
        base_status = status.split('-')[0].strip()  # Extract base status (CLEAR, STOP, WARNING)
        
        # Log detailed collision data to collision_log.csv
        with open(self.collision_log_path, 'a') as f:
            if not threats:
                f.write(f"{frame_idx},{timestamp},{base_status},none,inf,0.0\n")
            else:
                for threat in threats:
                    f.write(f"{frame_idx},{timestamp},{base_status},{threat['class_name']},"
                            f"{threat['distance']:.2f},{threat['roi_coverage']:.3f}\n")

        # Log base status to frame_status.csv
        with open(self.frame_status_log_path, 'a') as f:
            f.write(f"{frame_idx},{base_status}\n")

    def process_frame(self, frame, frame_idx):
        """Process a single frame with instance segmentation and depth analysis"""
        try:
            # Get depth map
            depth_map = self._get_depth_map(frame, frame_idx)
            
            # Run instance segmentation
            objects, seg_time = self.segmenter.detect(frame)
            
            # Calculate distance and status for each object
            status = "CLEAR"
            min_distance = float('inf')
            threats = []
            
            # Get ROI coordinates
            roi_coords = self.detector._dynamic_roi(frame.shape)
            x_s, x_e, y_s, y_e = roi_coords
            roi_area = (x_e - x_s) * (y_e - y_s)
            
            # Check for large objects (>90% of ROI area with at least 40% cross-section)
            large_obstacle_detected = False
            for obj_id, obj in objects.items():
                x1, y1, x2, y2 = obj['box']
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                
                # Calculate object area and cross-section with ROI
                obj_area = (x2 - x1) * (y2 - y1)
                mask = obj['mask']
                roi_mask = mask[y_s:y_e, x_s:x_e]
                overlap_area = np.sum(roi_mask)
                print("class = ",obj['class_name']," overlap_area = ",overlap_area)
                roi_coverage = overlap_area / roi_area if roi_area > 0 else 0.0
                obj_coverage = obj_area / roi_area if roi_area > 0 else 0.0
                
                if obj_coverage > self.large_size_threshold and roi_coverage >= self.min_cross_section_threshold:
                    distance = float('inf')  # Depth doesn't matter for large objects
                    status = f"STOP - {obj['class_name']} (Big Object)"
                    threats.append({
                        'class_name': obj['class_name'],
                        'distance': distance,
                        'mean_distance': float('inf'),
                        'score': obj['score'],
                        'roi_coverage': roi_coverage
                    })
                    large_obstacle_detected = True
                    break
            
            # If no large obstacle detected, proceed with depth estimation and unknown obstacles
            if not large_obstacle_detected:
                # Create a mask for detected objects
                detected_mask = np.zeros_like(depth_map, dtype=bool)
                for obj_id, obj in objects.items():
                    x1, y1, x2, y2 = obj['box']
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    detected_mask[y1:y2, x1:x2][obj['mask'][y1:y2, x1:x2] > 0] = True
                
                # Check for unknown obstacles in the upper 50% of ROI using depth map
                y_mid = y_s + (y_e - y_s) // 2  # Middle line of ROI
                roi_depth = depth_map[y_s:y_mid, x_s:x_e]  # Upper 50% of ROI
                undetected_mask = ~detected_mask[y_s:y_mid, x_s:x_e]
                unknown_depths = roi_depth[undetected_mask & (roi_depth > 0.01) & (roi_depth < 50.0)]
                
                if len(unknown_depths) > 0 and np.min(unknown_depths) < self.unknown_obstacle_threshold:
                    min_unknown_depth = float(np.percentile(unknown_depths, 3))
                    status = f"STOP - Unknown Obstacle ({min_unknown_depth:.1f}m)"
                    threats.append({
                        'class_name': 'Unknown Obstacle',
                        'distance': min_unknown_depth,
                        'mean_distance': float(np.mean(unknown_depths)),
                        'score': 1.0,
                        'roi_coverage': 1.0  # Placeholder, as we don't have a mask
                    })
                    # Compute bounding box for unknown obstacle (restricted to upper 50%)
                    y_idx, x_idx = np.where(undetected_mask & (roi_depth < self.unknown_obstacle_threshold))
                    if len(y_idx) > 0:
                        y_min, y_max = y_s + y_idx.min(), y_s + y_idx.max()
                        x_min, x_max = x_s + x_idx.min(), x_s + x_idx.max()
                        if (y_max-y_min)*(x_max-x_min)<0.2*(roi_area):
                            threats.pop()
                    else:
                        y_min, y_max = y_s, y_mid
                        x_min, x_max = x_s, x_e
                
                # Process remaining objects with depth estimation
                for obj_id, obj in objects.items():
                    x1, y1, x2, y2 = obj['box']
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    
                    # Check mask coverage in ROI
                    mask = obj['mask']
                    roi_mask = mask[y_s:y_e, x_s:x_e]
                    overlap_area = np.sum(roi_mask)
                    roi_coverage = overlap_area / roi_area if roi_area > 0 else 0.0
                    
                    # Extract depth values using mask
                    obj_depth = depth_map[y1:y2, x1:x2][obj['mask'][y1:y2, x1:x2] > 0]
                    valid_depths = obj_depth[(obj_depth > 0.0) & (obj_depth < 100.0)]
                    
                    if len(valid_depths) == 0:
                        obj['distance'] = float('inf')
                        print(f"Warning: No valid depths for {obj_id}")
                        with open(self.collision_log_path, 'a') as f:
                            f.write(f"{frame_idx},{time.time()},INVALID,{obj['class_name']},inf,0.0\n")
                        with open(self.frame_status_log_path, 'a') as f:
                            f.write(f"{frame_idx},INVALID\n")
                        continue
                    
                    distance = float(np.percentile(valid_depths, 5))
                    obj['distance'] = distance
                    
                    threat_data = {
                        'class_name': obj['class_name'],
                        'distance': distance,
                        'mean_distance': float(np.mean(valid_depths)),
                        'score': obj['score'],
                        'roi_coverage': roi_coverage
                    }
                    
                    # Update threats and status based on depth
                    if distance < self.warning_dist:
                        threats.append(threat_data)
                        if distance < self.stop_dist and roi_coverage >= self.roi_coverage_threshold and not large_obstacle_detected:
                            status = f"STOP - {obj['class_name']} ({distance:.1f}m)"
                        elif distance < self.warning_dist and status == "CLEAR" and not large_obstacle_detected:
                            status = f"WARNING - {obj['class_name']} ({distance:.1f}m)"
                    
                    if distance < min_distance:
                        min_distance = distance

            # Log collision data
            self._log_collision(frame_idx, status, threats)
            
            # Create visualization
            instance_vis = self.segmenter.visualize(frame, objects, depth_map)
            
            # Add unknown obstacle bounding box if detected
            if 'Unknown Obstacle' in [t['class_name'] for t in threats]:
                cv2.rectangle(instance_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(instance_vis, f"Unknown: {min_unknown_depth:.1f}m", 
                           (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add large obstacle bounding box if detected
            if large_obstacle_detected:
                for threat in threats:
                    if threat['roi_coverage'] >= self.min_cross_section_threshold:
                        x1, y1, x2, y2 = objects[list(objects.keys())[threats.index(threat)]]['box']
                        cv2.rectangle(instance_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(instance_vis, f"Big Object: {threat['distance']:.1f}m", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break
            
            # Add frame status at top right
            status_color = (0, 0, 255) if "STOP" in status else (0, 165, 255) if "WARNING" in status else (0, 255, 0)
            text = f"Frame: {frame_idx} | {status}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = frame.shape[1] - text_size[0] - 10  # Right-aligned, 10px from edge
            text_y = 30  # Top, 30px down
            cv2.putText(instance_vis, text, (text_x, text_y), font, font_scale, status_color, thickness)
            
            # Create depth visualization
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Add processing time info
            cv2.putText(depth_vis, f"Depth: {self.detector.avg_process_time*1000:.1f}ms | Segmentation: {seg_time*1000:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw ROI box in white on both visualizations
            cv2.rectangle(instance_vis, (x_s, y_s), (x_e, y_e), (255, 255, 255), 2)
            cv2.rectangle(depth_vis, (x_s, y_s), (x_e, y_e), (255, 255, 255), 2)
            
            # Combine visualizations
            combined = np.hstack((instance_vis, depth_vis))
            cv2.imwrite(os.path.join(self.output_dir, f"frame_{frame_idx:05d}.jpg"), combined)
            
            # Update progress
            self.processed_frames += 1
            self._display_progress()
            
            return combined, status
            
        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {str(e)}")
            with open(self.frame_status_log_path, 'a') as f:
                f.write(f"{frame_idx},ERROR\n")
            return frame, "ERROR"

    def _get_last_processed_frame(self):
        """Get the index of the last processed frame"""
        output_files = sorted([f for f in os.listdir(self.output_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        if not output_files:
            return -1
        last_file = output_files[-1]
        last_idx = int(last_file.split('_')[1].split('.')[0])
        return last_idx

    def run(self, interactive=False):
        """Run the navigation system on all frames"""
        self.start_time = time.time()
        
        if len([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')]) == 0:
            print("Extracting frames...")
            self.total_frames = self._extract_frames()
        else:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
            self.total_frames = len(frame_files)
        
        last_processed = self._get_last_processed_frame()
        self.processed_frames = last_processed + 1
        
        print(f"\nFound {self.processed_frames} processed frames. Total frames to process: {self.total_frames}")
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx <= last_processed:
                continue
            
            frame = cv2.imread(os.path.join(self.frames_dir, frame_file))
            if frame is None: 
                print(f"\nFailed to read frame {frame_file}")
                continue
            
            vis, status = self.process_frame(frame, frame_idx)
            
            if interactive:
                cv2.imshow('NavigationSystem', cv2.resize(vis, (1280, 480)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
        
        print("\n" + " " * 100 + "\r", end='')
        print(f"Processing complete. Total time: {timedelta(seconds=int(time.time() - self.start_time))}")
        
        # Call path_track3
        collision_log_path = os.path.join("output", "frame_status.csv")
        path_track.main(frames_folder=self.frames_dir, collision_log_path=collision_log_path)
        
        if interactive:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = EnhancedNavigationSystem("video.mp4")
    nav.run(interactive=False)
