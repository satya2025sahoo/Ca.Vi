import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
import pickle
import argparse
import shutil
import uuid

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

class VisualPathTracker:
    def __init__(self, cam, collision_log_path, checkpoint_dir="checkpoints", frame_rate=6, scale_factor=0.1):
        self.cam = cam
        self.K = cam.K
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21), 
                             maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.min_features = 1500
        self.frame_stage = 0
        self.last_frame = None
        self.px_ref = None
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []
        self.motion_history = deque(maxlen=5)
        self.turn_locations = []
        
        self.load_collision_log(collision_log_path)
        self.in_stop_mode = False
        self.last_valid_direction = np.zeros((3, 1))
        self.last_valid_rotation = np.eye(3)
        self.stop_counter = 0
        
        self.has_gt = False
        self.true_positions = []
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.latest_checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
        
        # Speed calculation parameters
        self.frame_rate = frame_rate  # Frames per second
        self.scale_factor = scale_factor  # Meters per pixel
        self.displacements = []  # Store displacements for speed calculation
        self.speed_intervals = []  # Store (frame_idx, speed) for each 60-frame interval
        self.interval_frames = 60  # Frames per 10 seconds (6 FPS * 10s)

    def load_collision_log(self, collision_log_path):
        try:
            self.collision_df = pd.read_csv(collision_log_path)
            self.status_list = self.collision_df['status'].tolist()
            print(f"Loaded collision log with {len(self.status_list)} entries")
        except Exception as e:
            print(f"Error loading collision log: {str(e)}")
            self.status_list = []

    def detect_features(self, frame):
        keypoints = self.detector.detect(frame)
        if len(keypoints) == 0:
            return np.array([])
        return np.array([x.pt for x in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    def track_features(self, prev_frame, curr_frame, prev_pts):
        if prev_pts is None or len(prev_pts) == 0:
            return np.array([]), np.array([])
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_pts, None, **self.lk_params)
        status = status.ravel()
        good_old = prev_pts[status == 1]
        good_new = curr_pts[status == 1]
        return good_old, good_new

    def estimate_motion(self, pts1, pts2):
        if len(pts1) < 8 or len(pts2) < 8:
            return None, None
        E, mask = cv2.findEssentialMat(
            pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K, mask=mask)
        return R, t

    def analyze_motion(self, R, t):
        if R is None or t is None:
            return "unknown", 0
        self.motion_history.append((R.copy(), t.copy()))
        cum_R = np.eye(3)
        for R_hist, _ in self.motion_history:
            cum_R = R_hist @ cum_R
        angle_axis, _ = cv2.Rodrigues(cum_R)
        angle = np.linalg.norm(angle_axis)
        angle_deg = np.degrees(angle)
        max_angle_change = 10.0
        if angle_deg > max_angle_change:
            angle_deg = min(angle_deg, max_angle_change)
        if angle_deg > 20:
            if angle_axis[1, 0] > 0:
                return "left", angle_deg
            else:
                return "right", angle_deg
        return "straight", 0

    def save_checkpoint(self, frame_idx):
        state = {
            'frame_stage': self.frame_stage,
            'cur_R': self.cur_R,
            'cur_t': self.cur_t,
            'trajectory': self.trajectory,
            'motion_history': list(self.motion_history),
            'turn_locations': self.turn_locations,
            'in_stop_mode': self.in_stop_mode,
            'last_valid_direction': self.last_valid_direction,
            'last_valid_rotation': self.last_valid_rotation,
            'stop_counter': self.stop_counter,
            'px_ref': self.px_ref,
            'last_frame': self.last_frame,
            'displacements': self.displacements,
            'speed_intervals': self.speed_intervals
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{frame_idx:05d}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        with open(self.latest_checkpoint_file, 'w') as f:
            f.write(checkpoint_path)
        traj_data = pd.DataFrame({
            'frame_idx': [frame_idx],
            'x': [self.cur_t[0, 0]],
            'y': [self.cur_t[1, 0]],
            'z': [self.cur_t[2, 0]],
            'status': [self.status_list[frame_idx] if frame_idx < len(self.status_list) else 'UNKNOWN']
        })
        traj_csv_path = os.path.join(self.checkpoint_dir, f"traj_{frame_idx:05d}.csv")
        traj_data.to_csv(traj_csv_path, index=False)

    def load_checkpoint(self):
        if not os.path.exists(self.latest_checkpoint_file):
            return None
        with open(self.latest_checkpoint_file, 'r') as f:
            checkpoint_path = f.read().strip()
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        self.frame_stage = state['frame_stage']
        self.cur_R = state['cur_R']
        self.cur_t = state['cur_t']
        self.trajectory = state['trajectory']
        self.motion_history = deque(state['motion_history'], maxlen=5)
        self.turn_locations = state['turn_locations']
        self.in_stop_mode = state['in_stop_mode']
        self.last_valid_direction = state['last_valid_direction']
        self.last_valid_rotation = state['last_valid_rotation']
        self.stop_counter = state['stop_counter']
        self.px_ref = state['px_ref']
        self.last_frame = state['last_frame']
        self.displacements = state.get('displacements', [])
        self.speed_intervals = state.get('speed_intervals', [])
        frame_idx = int(checkpoint_path.split('_')[-1].split('.')[0])
        return frame_idx

    def update(self, frame, frame_idx):
        current_is_stop = frame_idx < len(self.status_list) and self.status_list[frame_idx] == "STOP"
        
        if self.last_frame is None:
            self.last_frame = frame
            self.px_ref = self.detect_features(frame)
            self.frame_stage = 1
            self.save_checkpoint(frame_idx)
            return False
        
        displacement = np.zeros((3, 1))
        if current_is_stop:
            self.stop_counter += 1
            self.in_stop_mode = True
            if len(self.trajectory) >= 1:
                if self.stop_counter > 1:
                    displacement = np.zeros((3, 1))
                else:
                    norm = np.linalg.norm(self.cur_t)
                    if norm > 0.001:
                        last_dir = self.cur_t / norm
                        displacement = last_dir * 0.01
                    else:
                        displacement = np.array([[0], [0], [0.01]])
                self.cur_t = self.cur_t + displacement
                self.trajectory.append(self.cur_t.copy().flatten())
                self.displacements.append(np.linalg.norm(displacement) * self.scale_factor)
                self.save_checkpoint(frame_idx)
                return True
            self.save_checkpoint(frame_idx)
            return False
            
        if self.in_stop_mode:
            self.px_ref = self.detect_features(self.last_frame)
            self.in_stop_mode = False
            self.stop_counter = 0
        
        if self.frame_stage == 1:
            pts1, pts2 = self.track_features(self.last_frame, frame, self.px_ref)
            if len(pts1) < 8:
                self.px_ref = self.detect_features(self.last_frame)
                self.last_frame = frame
                self.save_checkpoint(frame_idx)
                return False
            self.cur_R, self.cur_t = self.estimate_motion(pts1, pts2)
            if self.cur_R is None:
                self.px_ref = self.detect_features(self.last_frame)
                self.last_frame = frame
                self.save_checkpoint(frame_idx)
                return False
            self.trajectory.append(self.cur_t.copy().flatten())
            self.displacements.append(np.linalg.norm(self.cur_t) * self.scale_factor)
            self.px_ref = self.detect_features(frame)
            self.frame_stage = 2
            
        elif self.frame_stage == 2:
            pts1, pts2 = self.track_features(self.last_frame, frame, self.px_ref)
            if len(pts1) < 8:
                self.px_ref = self.detect_features(self.last_frame)
                pts1, pts2 = self.track_features(self.last_frame, frame, self.px_ref)
                if len(pts1) < 8:
                    self.last_frame = frame
                    self.save_checkpoint(frame_idx)
                    return False
            R, t = self.estimate_motion(pts1, pts2)
            if R is None:
                self.px_ref = self.detect_features(self.last_frame)
                self.last_frame = frame
                self.save_checkpoint(frame_idx)
                return False
            if np.linalg.norm(t) > 0.01:
                self.last_valid_direction = t.copy()
                self.last_valid_rotation = R.copy()
            motion_type, angle = self.analyze_motion(R, t)
            if motion_type != "straight" and angle > 25:
                angle_factor = min(1.0, angle / 90.0)
                t = t * (0.7 + 0.3 * angle_factor)
                self.turn_locations.append((len(self.trajectory), motion_type, angle))
            if self.stop_counter > 0:
                self.stop_counter -= 1
                blend_factor = min(1.0, self.stop_counter / 5.0)
                t = blend_factor * self.last_valid_direction + (1 - blend_factor) * t
                R_blend = cv2.composeRT(self.last_valid_rotation, np.zeros((3,1)), 
                                       R, np.zeros((3,1)))[0]
                R = R_blend
            scale = 1.0
            displacement = scale * (self.cur_R @ t)
            self.cur_t = self.cur_t + displacement
            self.cur_R = R @ self.cur_R
            self.trajectory.append(self.cur_t.copy().flatten())
            self.displacements.append(np.linalg.norm(displacement) * self.scale_factor)
            if len(pts1) < self.min_features:
                self.px_ref = self.detect_features(frame)
            else:
                self.px_ref = self.detect_features(frame)
        
        # Calculate speed every 60 frames (10 seconds)
        if frame_idx % self.interval_frames == 0 and frame_idx > 0:
            interval_displacements = self.displacements[-self.interval_frames:]
            total_distance = sum(interval_displacements)
            time_interval = self.interval_frames / self.frame_rate  # e.g., 60 / 6 = 10 seconds
            speed = total_distance / time_interval if time_interval > 0 else 0  # Meters per second
            self.speed_intervals.append((frame_idx, speed))
        
        self.last_frame = frame
        self.save_checkpoint(frame_idx)
        return True

    def get_trajectory(self):
        return np.array(self.trajectory) if self.trajectory else np.array([np.zeros(3)])

    def smooth_trajectory(self):
        if len(self.trajectory) > 15:
            traj = np.array(self.trajectory)
            for i in range(3):
                window = min(15, len(traj) - 1)
                if window % 2 == 0:
                    window -= 1
                if window > 2:
                    traj[:,i] = savgol_filter(traj[:,i], window, 2)
            return traj
        return np.array(self.trajectory) if self.trajectory else np.array([np.zeros(3)])

def load_images(folder_path, max_frames=None):
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return images, []
    filenames = sorted([f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if max_frames is not None:
        filenames = filenames[:max_frames]
    print(f"Loading {len(filenames)} images from {folder_path}")
    for filename in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape[1] > 1280:
                scale = 1280 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            images.append(img)
    return images, filenames

def setup_visualization():
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Vehicle Trajectory')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Z Position')
    ax1.grid(True)
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Current Frame')
    plt.tight_layout()
    plt.ion()
    return ax1, ax2

def run_visual_odometry(frames_folder, collision_log_path, output_dir="path_track", checkpoint_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        collision_df = pd.read_csv(collision_log_path)
        max_frames = len(collision_df)
        print(f"Processing up to {max_frames} frames based on collision log")
    except Exception as e:
        print(f"Warning: Could not load collision log: {str(e)}")
        max_frames = None
    
    images, frame_files = load_images(frames_folder, max_frames)
    if len(images) < 2:
        print("Error: Need at least 2 frames for processing")
        return
    
    height, width = images[0].shape
    fx = width * 0.8
    fy = fx
    cx = width / 2
    cy = height / 2
    
    cam = PinholeCamera(width, height, fx, fy, cx, cy)
    tracker = VisualPathTracker(cam, collision_log_path, checkpoint_dir, frame_rate=6, scale_factor=0.1)
    
    start_idx = tracker.load_checkpoint()
    if start_idx is not None:
        print(f"Resuming from checkpoint at frame {start_idx}")
        start_idx += 1
    else:
        start_idx = 0
    
    ax1, ax2 = setup_visualization()
    traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='Path')
    current_pos, = ax1.plot([], [], 'ro', markersize=8, label='Current Position')
    start_marker, = ax1.plot([], [], 'ko', markersize=10, label='Start')
    turn_markers = ax1.scatter([], [], c='g', s=80, marker='^', label='Turns')
    ax1.legend()
    
    frame_display = ax2.imshow(images[0], cmap='gray')
    feature_points, = ax2.plot([], [], 'r.', markersize=3, alpha=0.5)
    
    last_saved_path = None
    try:
        for i in tqdm(range(start_idx, len(images)), desc="Processing frames", initial=start_idx, total=len(images)):
            try:
                frame_name = frame_files[i]
                frame_idx = int(''.join(filter(str.isdigit, frame_name)))
            except:
                frame_idx = i
            output_path = os.path.join(output_dir, f"path_{frame_idx:05d}.jpg")
            current_status = "CLEAR"
            if frame_idx < len(tracker.status_list):
                current_status = tracker.status_list[frame_idx]
            frame = images[i]
            result = tracker.update(frame, frame_idx)
            if not result and i > 0:
                if last_saved_path and os.path.exists(last_saved_path):
                    shutil.copy(last_saved_path, output_path)
                continue
            trajectory = tracker.get_trajectory()
            if len(trajectory) == 0:
                continue
            x_coords = [p[0] for p in trajectory]
            z_coords = [p[2] for p in trajectory]
            traj_line.set_data(x_coords, z_coords)
            current_pos.set_data([trajectory[-1][0]], [trajectory[-1][2]])
            if i == start_idx:
                start_marker.set_data([trajectory[0][0]], [trajectory[0][2]])
            if tracker.turn_locations:
                turn_x = [trajectory[idx][0] for idx, _, _ in tracker.turn_locations if idx < len(trajectory)]
                turn_z = [trajectory[idx][2] for idx, _, _ in tracker.turn_locations if idx < len(trajectory)]
                turn_markers.set_offsets(np.column_stack((turn_x, turn_z)))
            frame_display.set_array(frame)
            if tracker.px_ref is not None and len(tracker.px_ref) > 0:
                pts = tracker.px_ref.reshape(-1, 2)
                feature_points.set_data(pts[:, 0], pts[:, 1])
            if x_coords and z_coords:
                margin = max(5, 0.2 * max(np.ptp(x_coords), np.ptp(z_coords)))
                ax1.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
                ax1.set_ylim(min(z_coords)-margin, max(z_coords)+margin)
            
            # Annotate speed on image for CLEAR or WARNING frames at 60-frame intervals
            speed_text = ""
            if current_status in ["CLEAR", "WARNING"]:
                for idx, speed in tracker.speed_intervals:
                    if idx == frame_idx:
                        speed_text = f"Speed: {speed:.2f} m/s"
                        ax1.text(trajectory[-1][0], trajectory[-1][2], speed_text, fontsize=8, color='yellow', bbox=dict(facecolor='black', alpha=0.5))
                        break
            ax2.set_title(f"Frame {frame_idx}: {current_status}\n{speed_text}")
            plt.pause(0.001)
            plt.savefig(output_path, bbox_inches='tight')
            last_saved_path = output_path
    except KeyboardInterrupt:
        print(f"Processing interrupted at frame {i}. Checkpoint saved.")
        return
    
    smoothed_trajectory = tracker.smooth_trajectory()
    plt.figure(figsize=(10, 8))
    plt.plot(smoothed_trajectory[:, 0], smoothed_trajectory[:, 2], 'b-', linewidth=2, label='Smoothed Path')
    trajectory = np.array(trajectory)
    if len(trajectory) > 0:
        plt.plot(trajectory[:, 0], trajectory[:, 2], 'r--', alpha=0.5, linewidth=1, label='Raw Path')
    for idx, turn_type, angle in tracker.turn_locations:
        if idx < len(smoothed_trajectory):
            plt.plot(smoothed_trajectory[idx, 0], smoothed_trajectory[idx, 2], 'g^', markersize=8)
            plt.text(smoothed_trajectory[idx, 0], smoothed_trajectory[idx, 2], 
                    f"{turn_type}\n{angle:.1f}°", fontsize=8)
    # Annotate speeds on final trajectory plot
    for frame_idx, speed in tracker.speed_intervals:
        if frame_idx < len(trajectory):
            plt.text(trajectory[frame_idx][0], trajectory[frame_idx][2], 
                     f"Speed: {speed:.2f} m/s", fontsize=8, color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    plt.title('Complete Vehicle Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.grid(True)
    plt.legend()
    final_path = os.path.join(output_dir, "final_trajectory.jpg")
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"Final trajectory saved to {final_path}")
    traj_data = pd.DataFrame(smoothed_trajectory, columns=['x', 'y', 'z'])
    traj_csv_path = os.path.join(output_dir, "trajectory.csv")
    traj_data.to_csv(traj_csv_path, index=False)
    print(f"Trajectory data saved to {traj_csv_path}")
    plt.ioff()
    plt.show()

def main(frames_folder="frames", collision_log_path="output/frame_status.csv"):
    parser = argparse.ArgumentParser(description="Visual Path Tracking with Checkpointing and Speed Estimation")
    parser.add_argument('--frames_folder', default=frames_folder, help="Folder containing frame images")
    parser.add_argument('--collision_log', default=collision_log_path, help="Path to collision log CSV")
    parser.add_argument('--output_dir', default="path_track", help="Output directory for results")
    parser.add_argument('--checkpoint_dir', default="checkpoints", help="Directory for checkpoints")
    args = parser.parse_args()
    
    print(f"Starting visual path tracking...")
    print(f"Frames: {args.frames_folder}")
    print(f"Collision log: {args.collision_log}")
    print(f"Output: {args.output_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    
    run_visual_odometry(args.frames_folder, args.collision_log, args.output_dir, args.checkpoint_dir)

if __name__ == "__main__":
    main()

    def create_videos_from_folders(folder, fps=12, output_video_prefix="output_video"):
        # Get list of image files in the folder
        image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files:
            print(f"No images found in {folder}.")
            return

        print(f"Found {len(image_files)} images in {folder}")
        
        # Get the dimensions of the first image to use as reference
        first_image_path = os.path.join(folder, image_files[0])
        first_frame = cv2.imread(first_image_path)
        if first_frame is None:
            print(f"Error: Could not read the first image {first_image_path}")
            return
        
        reference_height, reference_width = first_frame.shape[:2]
        print(f"Using reference dimensions: {reference_width}x{reference_height}")

        # Define video writer
        video_name = os.path.join(folder, f"{output_video_prefix}_{os.path.basename(folder)}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, (reference_width, reference_height))

        # Process and write each image to the video
        skipped_count = 0
        for image_file in tqdm(image_files, desc=f"Creating video for {folder}"):
            image_path = os.path.join(folder, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"Error: Failed to read {image_path}")
                skipped_count += 1
                continue
                
            # Resize if dimensions don't match reference
            height, width = frame.shape[:2]
            if height != reference_height or width != reference_width:
                print(f"Resizing image {image_file} from {width}x{height} to {reference_width}x{reference_height}")
                frame = cv2.resize(frame, (reference_width, reference_height))
                
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()
        
        if skipped_count > 0:
            print(f"Warning: {skipped_count} images were skipped due to read errors")
            
        print(f"Video saved as {video_name} with {len(image_files) - skipped_count} frames at {fps} FPS")
        expected_duration = (len(image_files) - skipped_count) / fps
        print(f"Expected video duration: {expected_duration:.2f} seconds")

    # Call the function to create videos for both folders
    #create_videos_from_folders("output1", fps=12)
    create_videos_from_folders("path_track1", fps=12)
