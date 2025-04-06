import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
from collision_detector1 import EnhancedCollisionDetector
from semantic_segmenter1 import AccurateSegmenter

class NavigationSystem:
    def __init__(self, video_path="video.mp4", output_dir="output", frames_dir="frames", depth_dir="depth_maps"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_dir = frames_dir
        self.depth_dir = depth_dir
        
        self.detector = EnhancedCollisionDetector()
        self.segmenter = AccurateSegmenter()
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        
        # Progress tracking variables
        self.start_time = None
        self.processed_frames = 0
        self.total_frames = 0

    def _frames_exist(self):
        return len([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')]) > 0
    
    def _get_processed_frames(self):
        """Get set of frame indices that have already been processed"""
        processed = set()
        for f in os.listdir(self.output_dir):
            if f.startswith('frame_') and f.endswith('.jpg'):
                try:
                    frame_idx = int(f[6:-4])  # Extract number from 'frame_XXXXX.jpg'
                    processed.add(frame_idx)
                except ValueError:
                    continue
        return processed

    def _extract_frames(self, target_fps=6):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / target_fps))
        
        frame_idx = 0
        saved_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                cv2.imwrite(os.path.join(self.frames_dir, f"frame_{saved_count:05d}.jpg"), frame)
                saved_count += 1
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames at {target_fps} FPS")
        return saved_count

    def _get_depth_map(self, frame, frame_idx):
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

    def _get_progress_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        remaining_frames = self.total_frames - self.processed_frames
        eta = timedelta(seconds=int(remaining_frames / fps)) if fps > 0 else timedelta(0)
        progress = (self.processed_frames / self.total_frames) * 100
        
        return {
            'processed': self.processed_frames,
            'total': self.total_frames,
            'progress': progress,
            'fps': fps,
            'elapsed': timedelta(seconds=int(elapsed)),
            'eta': eta
        }

    def _display_progress(self, stats):
        progress_bar = f"[{'=' * int(stats['progress']/2):<50}] {stats['progress']:.1f}%"
        status_line = (f"Processed: {stats['processed']}/{stats['total']} | "
                      f"FPS: {stats['fps']:.1f} | "
                      f"Elapsed: {stats['elapsed']} | "
                      f"ETA: {stats['eta']}")
        print(f"\r{progress_bar} {status_line}", end='', flush=True)

    def _get_centroid(self, mask):
        """Calculate centroid of a mask"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None
            
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def process_frame(self, frame, frame_idx):
        try:
            # Semantic segmentation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            segmented = self.segmenter.segment(frame_rgb)
            masks = self.segmenter.get_masks(segmented)
            
            # Depth and collision detection
            depth_map = self._get_depth_map(frame, frame_idx)
            status, distance, _ = self.detector.check_collision(frame, frame_idx)
            
            # Create visualizations
            semantic_vis = self.segmenter.visualize(frame_rgb, segmented)
            semantic_vis = cv2.cvtColor(semantic_vis, cv2.COLOR_RGB2BGR)
            
            # Add class and distance text at detection locations (without black box)
            for class_name, (mask, min_distance) in masks.items():
                if class_name != 'background':
                    centroid = self._get_centroid(mask)
                    if centroid:
                        text = f"{class_name}: {min_distance:.1f}m"
                        # Draw white text with black outline for visibility
                        cv2.putText(semantic_vis, text, (centroid[0], centroid[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline (thicker)
                        cv2.putText(semantic_vis, text, (centroid[0], centroid[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
            
            # Rest of the visualization code remains the same...
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            h, w = frame.shape[:2]
            depth_vis_resized = cv2.resize(depth_vis, (w, h))
            
            # Add white ROI box to depth map
            roi_coords = self.detector._dynamic_roi(frame.shape)
            x_s, x_e, y_s, y_e = roi_coords
            cv2.rectangle(depth_vis_resized, (x_s, y_s), (x_e, y_e), (255, 255, 255), 3)
            
            # Add status text to depth map
            status_text = f"{status} | {distance:.1f}m"
            cv2.putText(depth_vis_resized, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine views
            combined = np.hstack((semantic_vis, depth_vis_resized))
            cv2.imwrite(os.path.join(self.output_dir, f"frame_{frame_idx:05d}.jpg"), combined)
            
            # Update progress
            self.processed_frames += 1
            stats = self._get_progress_stats()
            self._display_progress(stats)
            
            return combined, status
            
        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {str(e)}")
            return frame, "ERROR"
    def run(self):
        # Initialize progress tracking
        self.start_time = time.time()
        
        if not self._frames_exist():
            print("Extracting frames...")
            self.total_frames = self._extract_frames()
        else:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
            self.total_frames = len(frame_files)
        
        # Get already processed frames
        processed_frames = self._get_processed_frames()
        initial_processed = len(processed_frames)
        self.processed_frames = initial_processed
        
        print(f"\nFound {initial_processed} already processed frames. Resuming processing...")
        print(f"Processing {self.total_frames - initial_processed} remaining frames out of {self.total_frames} total...")
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        for frame_idx, frame_file in enumerate(frame_files):
            # Skip already processed frames
            if frame_idx in processed_frames:
                continue
                
            frame = cv2.imread(os.path.join(self.frames_dir, frame_file))
            if frame is None: 
                print(f"\nFailed to read frame {frame_file}")
                continue
            
            vis, status = self.process_frame(frame, frame_idx)
            cv2.imshow('Navigation System', vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clear progress line and print completion
        print("\n" + " " * 100 + "\r", end='')  # Clear progress line
        total_time = timedelta(seconds=int(time.time() - self.start_time))
        print(f"Processing complete. Total time: {total_time}")
        print(f"Processed {self.processed_frames - initial_processed} new frames (total {self.processed_frames}/{self.total_frames})")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = NavigationSystem("test_video.mp4")
    nav.run()
