import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
from scipy.signal import savgol_filter
from collections import deque

# Camera parameters
image_width, image_height = 1920, 1080
focal_length_mm = 24
sensor_width_mm = 36

# Compute focal length in pixels
fx = (focal_length_mm / sensor_width_mm) * image_width
fy = fx
cx, cy = image_width / 2, image_height / 2

# Intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Feature tracking parameters
FAST_THRESHOLD = 20
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

# Motion model parameters
STRAIGHT_THRESHOLD = 0.1
TURN_THRESHOLD = 0.5
SMOOTHING_WINDOW = 7
MIN_TURN_ANGLE = 15  # degrees
MIN_OBSTACLE_ANGLE = 30  # degrees for obstacle avoidance
TURN_MEMORY = 5
OBSTACLE_MEMORY = 3  # Shorter memory for quick maneuvers

def load_images(folder_path):
    """Load images with validation"""
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return images
    
    filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
    for filename in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape[1] > 1280:
                scale = 1280 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            images.append(img)
    
    return images

def detect_features(img):
    """Feature detection with focus on road and static background"""
    if img is None or len(img.shape) != 2:
        return np.array([])
    
    fast = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    height, width = img.shape
    
    # Road region (lower 50%)
    road_roi = img[int(height*0.5):height, :]
    road_kps = fast.detect(road_roi, None)
    
    # Static background (upper 30%, excluding middle 20% which may contain moving objects)
    bg_roi = img[0:int(height*0.3), :]
    bg_kps = fast.detect(bg_roi, None)
    
    # Adjust keypoint coordinates
    for kp in road_kps:
        kp.pt = (kp.pt[0], kp.pt[1] + height*0.5)
    
    # Combine features (prioritize road features)
    all_kps = road_kps + bg_kps[:len(road_kps)//3]  # 3:1 ratio
    
    return cv2.KeyPoint_convert(all_kps) if all_kps else np.array([])

def track_features(img1, img2, points1):
    """Robust feature tracking with flow validation"""
    if img1 is None or img2 is None or len(points1) == 0:
        return np.array([]), np.array([])
    
    points2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **LK_PARAMS)
    if points2 is None:
        return np.array([]), np.array([])
    
    status = status.squeeze()
    valid = (status == 1) & (points1[:, 0] >= 0) & (points1[:, 1] >= 0) & \
            (points2[:, 0] >= 0) & (points2[:, 1] >= 0)
    
    if sum(valid) < 10:
        return np.array([]), np.array([])
    
    # Geometric verification
    F, mask = cv2.findFundamentalMat(points1[valid], points2[valid], cv2.FM_RANSAC, 1.0)
    if F is not None and mask is not None:
        valid[valid] = mask.squeeze().astype(bool)
    
    return points1[valid], points2[valid]

def analyze_motion(R, t, motion_history, obstacle_history):
    """Enhanced motion analysis with obstacle detection"""
    if R is None or t is None:
        return "straight", 0, False
    
    # Update histories
    motion_history.append((R.copy(), t.copy()))
    obstacle_history.append((R.copy(), t.copy()))
    
    # Regular turn detection (longer-term)
    cum_R = np.eye(3)
    for R_hist, _ in motion_history:
        cum_R = R_hist @ cum_R
    
    # Obstacle avoidance detection (shorter-term)
    obs_cum_R = np.eye(3)
    for R_hist, _ in obstacle_history:
        obs_cum_R = R_hist @ obs_cum_R
    
    # Convert to axis-angle
    angle_axis, _ = cv2.Rodrigues(cum_R)
    obs_angle_axis, _ = cv2.Rodrigues(obs_cum_R)
    
    angle = np.linalg.norm(angle_axis)
    obs_angle = np.linalg.norm(obs_angle_axis)
    
    cum_angle_deg = np.degrees(angle)
    obs_angle_deg = np.degrees(obs_angle)
    
    # Check for obstacle avoidance (sharp, quick turns)
    is_obstacle = False
    if obs_angle_deg > MIN_OBSTACLE_ANGLE:
        is_obstacle = True
        if obs_angle_axis[1, 0] > 0:
            return "left", obs_angle_deg, is_obstacle
        else:
            return "right", obs_angle_deg, is_obstacle
    
    # Regular turn detection
    if cum_angle_deg > MIN_TURN_ANGLE:
        if angle_axis[1, 0] > 0:
            return "left", cum_angle_deg, is_obstacle
        else:
            return "right", cum_angle_deg, is_obstacle
    
    return "straight", 0, is_obstacle

def estimate_trajectory(images, K):
    """Main trajectory estimation with real-time visualization"""
    if len(images) < 2:
        return np.array([]), [], []
    
    # Initialize visualization
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Live Car Trajectory')
    ax1.set_xlabel('X Position (meters)')
    ax1.set_ylabel('Z Position (meters)')
    ax1.grid(True)
    
    # Initialize trajectory plot elements
    traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='Path')
    current_pos, = ax1.plot([], [], 'ro', markersize=8, label='Current Position')
    start_marker, = ax1.plot([], [], 'ko', markersize=10, label='Start')
    turn_markers = ax1.scatter([], [], c='g', s=80, marker='^', label='Turns')
    obstacle_markers = ax1.scatter([], [], c='r', s=80, marker='x', label='Obstacles')
    ax1.legend()
    
    # Initialize frame display
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Current Frame')
    frame_display = ax2.imshow(images[0], cmap='gray')
    feature_display, = ax2.plot([], [], 'r.', markersize=3, alpha=0.5)
    
    plt.tight_layout()
    plt.ion()  # Interactive mode on
    plt.show()
    
    # Initialize tracking variables
    trajectory = []
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    motion_history = deque(maxlen=TURN_MEMORY)
    obstacle_history = deque(maxlen=OBSTACLE_MEMORY)
    turn_info = []
    obstacle_info = []
    turn_coords = []
    obstacle_coords = []
    
    prev_img = images[0]
    prev_points = detect_features(prev_img)
    
    for i in tqdm(range(len(images)-1), desc="Processing frames"):
        curr_img = images[i+1]
        
        # Track features between frames
        prev_points, curr_points = track_features(prev_img, curr_img, prev_points)
        
        # Adaptive feature re-detection if needed
        if len(prev_points) < 1000:
            prev_points = detect_features(prev_img)
            if len(prev_points) == 0:
                continue
            prev_points, curr_points = track_features(prev_img, curr_img, prev_points)
            if len(prev_points) == 0:
                continue
        
        # Estimate camera motion
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue
            
        _, R, t, _ = cv2.recoverPose(E, curr_points, prev_points, K, mask=mask)
        
        # Analyze motion type
        motion_type, angle, is_obstacle = analyze_motion(R, t, motion_history, obstacle_history)
        
        # Apply motion constraints based on motion type
        if motion_type == "straight":
            t[0] *= 0.2  # Reduce lateral drift
            t[1] *= 0.1  # Reduce vertical motion
        elif is_obstacle:
            t[0] *= 2.5  # Stronger lateral motion for obstacles
            obstacle_coords.append((t_total[0,0], t_total[2,0]))
            obstacle_info.append((i, motion_type, angle))
        else:
            turn_factor = min(2.0, 1 + angle / 45)
            t[0] *= turn_factor
            turn_coords.append((t_total[0,0], t_total[2,0]))
            turn_info.append((i, motion_type, angle))
        
        # Update global pose
        t_total = t_total + R_total @ t
        R_total = R @ R_total
        trajectory.append(t_total.copy().flatten())
        
        # Update visualization
        x_coords = [p[0] for p in trajectory]
        z_coords = [p[2] for p in trajectory]
        
        traj_line.set_data(x_coords, z_coords)
        current_pos.set_data([t_total[0,0]], [t_total[2,0]])
        
        if i == 0:
            start_marker.set_data([t_total[0,0]], [t_total[2,0]])
        
        if turn_coords:
            turn_markers.set_offsets(turn_coords)
        if obstacle_coords:
            obstacle_markers.set_offsets(obstacle_coords)
        
        # Update frame display
        frame_display.set_array(curr_img)
        if len(curr_points) > 0:
            feature_display.set_data(curr_points[:,0], curr_points[:,1])
        
        # Adjust view limits
        if x_coords and z_coords:
            margin = max(5, 0.2 * max(np.ptp(x_coords), np.ptp(z_coords)))
            ax1.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
            ax1.set_ylim(min(z_coords)-margin, max(z_coords)+margin)
        
        plt.pause(0.001)  # Small pause to update display
        
        # Prepare for next frame
        prev_img = curr_img
        prev_points = curr_points
    
    plt.ioff()  # Turn off interactive mode
    
    # Post-processing smoothing
    if len(trajectory) > SMOOTHING_WINDOW:
        trajectory = np.array(trajectory)
        for i in range(3):
            trajectory[:,i] = savgol_filter(trajectory[:,i], SMOOTHING_WINDOW, 2)
    else:
        trajectory = np.array(trajectory)
    
    return trajectory, turn_info, obstacle_info

def main():
    # Specify your frames folder path
    frames_folder = "frames"  # Update this path
    
    print("\n=== Loading Frames ===")
    images = load_images(frames_folder)
    if len(images) < 2:
        print("Need at least 2 frames for processing. Exiting.")
        return
    
    print("\n=== Estimating Trajectory ===")
    start_time = time.time()
    trajectory, turn_info, obstacle_info = estimate_trajectory(images, K)
    end_time = time.time()
    
    if len(trajectory) == 0:
        print("Trajectory estimation failed. Exiting.")
        return
    
    print(f"\nProcessing completed in {end_time-start_time:.2f} seconds")
    print(f"Trajectory points: {len(trajectory)}")
    print(f"Turns detected: {len(turn_info)}")
    print(f"Obstacle maneuvers: {len(obstacle_info)}")
    
    # Keep the final visualization window open
    plt.show(block=True)

if __name__ == "__main__":
    main()