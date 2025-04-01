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

    def process_frame(self, frame, frame_idx):
        try:
            # Semantic segmentation
            segmented = self.segmenter.segment(frame)
            masks = self.segmenter.get_masks(segmented)
            
            # Depth and collision detection
            depth_map = self._get_depth_map(frame, frame_idx)
            status, distance, _ = self.detector.check_collision(frame, frame_idx, masks)
            
            # Visualizations
            collision_vis = self.detector.visualize(frame, depth_map, status.split('-')[0].strip(), distance)
            semantic_vis = self.segmenter.visualize(frame, segmented)
            
            # Combine views
            combined = np.hstack((collision_vis, semantic_vis))
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
        
        print(f"\nProcessing {self.total_frames} frames...")
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        for frame_idx, frame_file in enumerate(frame_files):
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
        print(f"Processing complete. Total time: {timedelta(seconds=int(time.time() - self.start_time))}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    nav = NavigationSystem("test_video.mp4")
    nav.run()