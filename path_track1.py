import cv2
import numpy as np
from collections import deque

class PathTracker:
    def __init__(self):
        # Camera intrinsics (example values - calibrate for your camera)
        self.fx = 1380  # Focal length in pixels
        self.fy = 1380
        self.cx = 960   # Principal point
        self.cy = 540
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
        self.last_known_good = None                # For obstacle fallback
        
        # Parameters
        self.MIN_FEATURES = 100
        self.MIN_TURN_ANGLE = 15    # degrees
        self.OBSTACLE_ANGLE = 30    # degrees

    def reset(self):
        """Reset tracking state for new video"""
        self.prev_img = None
        self.prev_points = None
        self.trajectory = []
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.motion_history.clear()
        self.obstacle_history.clear()
        self.last_known_good = None

    def visualize_path(self, frame_shape):
        """Generate standalone path visualization"""
        h, w = frame_shape[:2]
        path_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.trajectory) < 2:
            return path_img
        
        # Convert trajectory to image coordinates
        points = []
        for pose in self.trajectory:
            x = int(pose[0] * 50 + w//4)
            y = int(pose[2] * 50 + h//2)
            points.append((x, y))
        
        # Draw path
        for i in range(1, len(points)):
            color = (0, 0, 255) if self._was_obstacle(i) else (255, 0, 255)
            cv2.line(path_img, points[i-1], points[i], color, 2)
        
        return path_img

    def _was_obstacle(self, index):
        """Check if frame had obstacle (simplified)"""
        if index >= len(self.trajectory):
            return False
        # Add your obstacle detection logic here
        return False

    def update(self, frame, collision_status="CLEAR"):
        """
        Main update method
        Args:
            frame: Current video frame (BGR)
            collision_status: From collision detector ("CLEAR"/"WARNING"/"STOP")
        Returns:
            dict: {"direction": "left"/"right"/"straight", 
                  "angle": degrees,
                  "is_obstacle": bool}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self.prev_img is None:
            self.prev_img = gray
            self.prev_points = self._detect_features(gray)
            return {"direction": "straight", "angle": 0, "is_obstacle": False}
        
        # Handle obstacle cases
        if collision_status != "CLEAR":
            result = self._handle_obstacle(gray)
            result["is_obstacle"] = True
            return result
            
        # Normal feature tracking
        return self._track_frame(gray)

    def _track_frame(self, gray):
        """Standard visual odometry tracking"""
        # Track features
        curr_points, self.prev_points = self._track_features(
            self.prev_img, gray, self.prev_points
        )
        
        # Re-detect if too few features
        if len(curr_points) < self.MIN_FEATURES:
            self.prev_points = self._detect_features(gray)
            return {"direction": "straight", "angle": 0, "is_obstacle": False}
        
        # Estimate motion
        E, mask = cv2.findEssentialMat(
            curr_points, self.prev_points, self.K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        _, R, t, _ = cv2.recoverPose(E, curr_points, self.prev_points, self.K, mask=mask)
        
        # Analyze motion
        motion_type, angle, is_obstacle = self._analyze_motion(R, t)
        
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
        
        # Continue with dampened last known motion
        dampening = 0.3  # Reduce motion magnitude during obstacles
        self.t_total += self.R_total @ (self.last_known_good["t"] * dampening)
        self.R_total = self.last_known_good["R"] @ self.R_total
        self.trajectory.append(self.t_total.copy().flatten())
        
        # Try to maintain some feature tracking
        self.prev_points = self._track_features(
            self.prev_img, gray, self.last_known_good["points"]
        )[0]
        self.prev_img = gray
        
        return {"direction": "straight", "angle": 0}

    def _detect_features(self, img):
        """Detect features with road priority"""
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        height, width = img.shape
        
        # Road features (lower 50%)
        road_roi = img[int(height*0.5):, :]
        road_kps = fast.detect(road_roi, None)
        
        # Adjust coordinates
        for kp in road_kps:
            kp.pt = (kp.pt[0], kp.pt[1] + height*0.5)
        
        return cv2.KeyPoint_convert(road_kps)

    def _track_features(self, img1, img2, pts1):
        """KLTDense optical flow tracking"""
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts1, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Validate points
        if pts2 is None:
            return np.array([]), np.array([])
        
        status = status.squeeze()
        valid = status == 1
        return pts2[valid], pts1[valid]

    def _analyze_motion(self, R, t):
        """Determine motion type and obstacle flags"""
        # Update history
        self.motion_history.append((R.copy(), t.copy()))
        self.obstacle_history.append((R.copy(), t.copy()))
        
        # Cumulative rotation analysis
        cum_R = np.eye(3)
        for R_hist, _ in self.motion_history:
            cum_R = R_hist @ cum_R
        
        # Obstacle rotation analysis (shorter window)
        obs_cum_R = np.eye(3)
        for R_hist, _ in self.obstacle_history:
            obs_cum_R = R_hist @ obs_cum_R
        
        # Convert to axis-angle
        angle_axis, _ = cv2.Rodrigues(cum_R)
        obs_angle_axis, _ = cv2.Rodrigues(obs_cum_R)
        
        angle = np.degrees(np.linalg.norm(angle_axis))
        obs_angle = np.degrees(np.linalg.norm(obs_angle_axis))
        
        # Check for obstacle avoidance
        if obs_angle > self.OBSTACLE_ANGLE:
            direction = "left" if obs_angle_axis[1,0] > 0 else "right"
            return direction, obs_angle, True
        
        # Check normal turns
        if angle > self.MIN_TURN_ANGLE:
            direction = "left" if angle_axis[1,0] > 0 else "right"
            return direction, angle, False
            
        return "straight", 0, False

    def get_trajectory(self):
        """Get current trajectory as numpy array"""
        return np.array(self.trajectory) if self.trajectory else np.array([])