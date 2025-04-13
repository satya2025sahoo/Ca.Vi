import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class PathTracker:
    def __init__(self):
        # Camera intrinsics (aligned with Code 1 for consistency)
        self.image_width, self.image_height = 1920, 1080
        self.focal_length_mm = 24
        self.sensor_width_mm = 36
        self.fx = (self.focal_length_mm / self.sensor_width_mm) * self.image_width
        self.fy = self.fx
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]])
        
        # Tracking state
        self.prev_img = None
        self.prev_points = None
        self.trajectory = []
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        
        # Motion analysis
        self.motion_history = deque(maxlen=5)      # For turn detection
        self.obstacle_history = deque(maxlen=3)    # For obstacle detection
        self.last_known_good = None
        self.turn_info = []                       # From Code 1: Track turns
        self.obstacle_info = []                   # From Code 1: Track obstacles
        
        # Parameters
        self.MIN_FEATURES = 100
        self.MIN_TURN_ANGLE = 15
        self.OBSTACLE_ANGLE = 30
        self.FAST_THRESHOLD = 20
        self.LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        self.SMOOTHING_WINDOW = 7                 # From Code 1
        
        # Visualization (from Code 1)
        self.visualize_enabled = False
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.traj_line = None
        self.current_pos = None
        self.start_marker = None
        self.turn_markers = None
        self.obstacle_markers = None
        self.frame_display = None
        self.feature_display = None

    def enable_visualization(self):
        """Initialize Matplotlib visualization (from Code 1)"""
        self.visualize_enabled = True
        self.fig = plt.figure(figsize=(14, 6))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax1.set_title('Live Car Trajectory')
        self.ax1.set_xlabel('X Position (meters)')
        self.ax1.set_ylabel('Z Position (meters)')
        self.ax1.grid(True)
        
        self.traj_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Path')
        self.current_pos, = self.ax1.plot([], [], 'ro', markersize=8, label='Current Position')
        self.start_marker, = self.ax1.plot([], [], 'ko', markersize=10, label='Start')
        self.turn_markers = self.ax1.scatter([], [], c='g', s=80, marker='^', label='Turns')
        self.obstacle_markers = self.ax1.scatter([], [], c='r', s=80, marker='x', label='Obstacles')
        self.ax1.legend()
        
        self.ax2 = plt.subplot(1, 2, 2)
        self.ax2.set_title('Current Frame')
        self.frame_display = self.ax2.imshow(np.zeros((self.image_height, self.image_width)), cmap='gray')
        self.feature_display, = self.ax2.plot([], [], 'r.', markersize=3, alpha=0.5)
        
        plt.tight_layout()
        plt.ion()

    def reset(self):
        """Reset tracking state"""
        self.prev_img = None
        self.prev_points = None
        self.trajectory = []
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.motion_history.clear()
        self.obstacle_history.clear()
        self.last_known_good = None
        self.turn_info = []
        self.obstacle_info = []
        if self.visualize_enabled:
            self.traj_line.set_data([], [])
            self.current_pos.set_data([], [])
            self.start_marker.set_data([], [])
            self.turn_markers.set_offsets([])
            self.obstacle_markers.set_offsets([])
            self.feature_display.set_data([], [])

    def visualize_path(self, frame_shape):
        """Generate standalone path visualization (retained from Code 2)"""
        h, w = frame_shape[:2]
        path_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            return path_img
        
        points = []
        for pose in self.trajectory:
            x = int(pose[0] * 50 + w//4)
            y = int(pose[2] * 50 + h//2)
            points.append((x, y))
        
        for i in range(1, len(points)):
            color = (0, 0, 255) if self._was_obstacle(i) else (255, 0, 255)
            cv2.line(path_img, points[i-1], points[i], color, 2)
        
        return path_img

    def _was_obstacle(self, index):
        """Check if frame had obstacle"""
        return any(info[0] == index for info in self.obstacle_info)

    def update(self, frame, collision_status="CLEAR", frame_index=0):
        """
        Main update method
        Args:
            frame: Current video frame (BGR)
            collision_status: "CLEAR"/"WARNING"/"STOP"
            frame_index: Frame number for tracking turn/obstacle info
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_img is None:
            self.prev_img = gray
            self.prev_points = self._detect_features(gray)
            if self.visualize_enabled:
                self.frame_display.set_array(gray)
                plt.pause(0.001)
            return {"direction": "straight", "angle": 0, "is_obstacle": False}
        
        if collision_status != "CLEAR":
            result = self._handle_obstacle(gray)
            result["is_obstacle"] = True
            self.obstacle_info.append((frame_index, result["direction"], result["angle"]))
            return result
            
        return self._track_frame(gray, frame_index)

    def _track_frame(self, gray, frame_index):
        """Standard visual odometry tracking"""
        curr_points, self.prev_points = self._track_features(self.prev_img, gray, self.prev_points)
        
        if len(curr_points) < self.MIN_FEATURES:
            self.prev_points = self._detect_features(gray)
            if self.visualize_enabled:
                self.frame_display.set_array(gray)
                plt.pause(0.001)
            return {"direction": "straight", "angle": 0, "is_obstacle": False}
        
        E, mask = cv2.findEssentialMat(
            curr_points, self.prev_points, self.K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            self.prev_points = self._detect_features(gray)
            return {"direction": "straight", "angle": 0, "is_obstacle": False}
            
        _, R, t, _ = cv2.recoverPose(E, curr_points, self.prev_points, self.K, mask=mask)
        
        # Apply motion constraints (from Code 1)
        motion_type, angle, is_obstacle = self._analyze_motion(R, t, frame_index)
        if motion_type == "straight":
            t[0] *= 0.2  # Reduce lateral drift
            t[1] *= 0.1  # Reduce vertical motion
        elif is_obstacle:
            t[0] *= 2.5  # Stronger lateral motion
        else:
            turn_factor = min(2.0, 1 + angle / 45)
            t[0] *= turn_factor
        
        # Update trajectory
        self.t_total += self.R_total @ t
        self.R_total = R @ self.R_total
        self.trajectory.append(self.t_total.copy().flatten())
        
        # Store last known good state
        if not is_obstacle:
            self.last_known_good = {
                "R": self.R_total.copy(),
                "t": self.t_total.copy(),
                "points": curr_points.copy()
            }
        
        # Update visualization
        if self.visualize_enabled:
            x_coords = [p[0] for p in self.trajectory]
            z_coords = [p[2] for p in self.trajectory]
            self.traj_line.set_data(x_coords, z_coords)
            self.current_pos.set_data([self.t_total[0,0]], [self.t_total[2,0]])
            
            if len(self.trajectory) == 1:
                self.start_marker.set_data([self.t_total[0,0]], [self.t_total[2,0]])
            
            turn_coords = [(self.trajectory[i][0], self.trajectory[i][2]) for i, _, _ in self.turn_info]
            obstacle_coords = [(self.trajectory[i][0], self.trajectory[i][2]) for i, _, _ in self.obstacle_info]
            self.turn_markers.set_offsets(turn_coords)
            self.obstacle_markers.set_offsets(obstacle_coords)
            
            self.frame_display.set_array(gray)
            if len(curr_points) > 0:
                self.feature_display.set_data(curr_points[:,0], curr_points[:,1])
            
            if x_coords and z_coords:
                margin = max(5, 0.2 * max(np.ptp(x_coords), np.ptp(z_coords)))
                self.ax1.set_xlim(min(x_coords)-margin, max(x_coords)+margin)
                self.ax1.set_ylim(min(z_coords)-margin, max(z_coords)+margin)
            
            plt.pause(0.001)
        
        # Update state
        self.prev_img = gray
        self.prev_points = curr_points
        
        return {
            "direction": motion_type,
            "angle": angle,
            "is_obstacle": is_obstacle
        }

    def _handle_obstacle(self, gray):
        """Fallback tracking when obstacle is detected"""
        if self.last_known_good is None:
            return {"direction": "straight", "angle": 0}
        
        dampening = 0.3
        self.t_total += self.R_total @ (self.last_known_good["t"] * dampening)
        self.R_total = self.last_known_good["R"] @ self.R_total
        self.trajectory.append(self.t_total.copy().flatten())
        
        self.prev_points = self._track_features(
            self.prev_img, gray, self.last_known_good["points"]
        )[0]
        self.prev_img = gray
        
        if self.visualize_enabled:
            self.frame_display.set_array(gray)
            plt.pause(0.001)
        
        return {"direction": "straight", "angle": 0}

    def _detect_features(self, img):
        """Detect features with road and background priority (from Code 1)"""
        fast = cv2.FastFeatureDetector_create(threshold=self.FAST_THRESHOLD, nonmaxSuppression=True)
        height, width = img.shape
        
        # Road features (lower 50%)
        road_roi = img[int(height*0.5):, :]
        road_kps = fast.detect(road_roi, None)
        for kp in road_kps:
            kp.pt = (kp.pt[0], kp.pt[1] + height*0.5)
        
        # Background features (upper 30%)
        bg_roi = img[0:int(height*0.3), :]
        bg_kps = fast.detect(bg_roi, None)
        
        # Combine (3:1 road to background ratio)
        all_kps = road_kps + bg_kps[:len(road_kps)//3]
        return cv2.KeyPoint_convert(all_kps) if all_kps else np.array([])

    def _track_features(self, img1, img2, pts1):
        """KLTDense optical flow with geometric verification (from Code 1)"""
        if len(pts1) == 0:
            return np.array([]), np.array([])
        
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts1, None, **self.LK_PARAMS
        )
        
        if pts2 is None:
            return np.array([]), np.array([])
        
        status = status.squeeze()
        valid = (status == 1) & (pts1[:, 0] >= 0) & (pts1[:, 1] >= 0) & \
                (pts2[:, 0] >= 0) & (pts2[:, 1] >= 0)
        
        if sum(valid) < 10:
            return np.array([]), np.array([])
        
        # Geometric verification
        F, mask = cv2.findFundamentalMat(pts1[valid], pts2[valid], cv2.FM_RANSAC, 1.0)
        if F is not None and mask is not None:
            valid[valid] = mask.squeeze().astype(bool)
        
        return pts2[valid], pts1[valid]

    def _analyze_motion(self, R, t, frame_index):
        """Determine motion type and obstacle flags"""
        self.motion_history.append((R.copy(), t.copy()))
        self.obstacle_history.append((R.copy(), t.copy()))
        
        cum_R = np.eye(3)
        for R_hist, _ in self.motion_history:
            cum_R = R_hist @ cum_R
        
        obs_cum_R = np.eye(3)
        for R_hist, _ in self.obstacle_history:
            obs_cum_R = R_hist @ obs_cum_R
        
        angle_axis, _ = cv2.Rodrigues(cum_R)
        obs_angle_axis, _ = cv2.Rodrigues(obs_cum_R)
        
        angle = np.degrees(np.linalg.norm(angle_axis))
        obs_angle = np.degrees(np.linalg.norm(obs_angle_axis))
        
        if obs_angle > self.OBSTACLE_ANGLE:
            direction = "left" if obs_angle_axis[1,0] > 0 else "right"
            self.obstacle_info.append((frame_index, direction, obs_angle))
            return direction, obs_angle, True
        
        if angle > self.MIN_TURN_ANGLE:
            direction = "left" if angle_axis[1,0] > 0 else "right"
            self.turn_info.append((frame_index, direction, angle))
            return direction, angle, False
            
        return "straight", 0, False

    def get_trajectory(self):
        """Get smoothed trajectory (from Code 1)"""
        traj = np.array(self.trajectory) if self.trajectory else np.array([])
        if len(traj) > self.SMOOTHING_WINDOW:
            smoothed = np.zeros_like(traj)
            for i in range(3):
                smoothed[:, i] = savgol_filter(traj[:, i], self.SMOOTHING_WINDOW, 2)
            return smoothed
        return traj

    def get_motion_info(self):
        """Return turn and obstacle information (from Code 1)"""
        return {"turns": self.turn_info, "obstacles": self.obstacle_info}

    def close(self):
        """Clean up visualization"""
        if self.visualize_enabled:
            plt.ioff()
            plt.close(self.fig)

