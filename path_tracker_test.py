import cv2
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import os
from torchvision.transforms import Compose, ToTensor, Normalize

class RobustCameraPathTracker:
    def __init__(self, frames_folder="frames", depth_folder="depth_maps", focal_length_mm=24, sensor_width_mm=36):
        # Camera intrinsics
        self.image_width, self.image_height = 1920, 1080  # Adjust based on your frames
        self.fx = (focal_length_mm / sensor_width_mm) * self.image_width
        self.fy = self.fx
        self.cx, self.cy = self.image_width / 2, self.image_height / 2
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]])

        # Depth model (Depth Anything V2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = self._load_depth_model()
        self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])

        # Tracking state
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.trajectory = [self.t_total.flatten()]
        self.prev_frame = None
        self.prev_points = None
        self.state = "MOVING"  # MOVING, STOPPED, OCCLUDED
        self.depth_history = deque(maxlen=5)  # For depth consistency
        self.last_known_good = None

        # Parameters
        self.MIN_FEATURES = 50
        self.DEPTH_THRESHOLD = 2.0  # Meters, sudden depth drop threshold
        self.OCCLUSION_COVERAGE = 0.8  # 80% of frame covered by close object
        self.MOTION_THRESHOLD = 0.05  # Translation magnitude threshold (tuned for 6 FPS)
        self.frames_folder = frames_folder
        self.depth_folder = depth_folder

        # Ensure depth folder exists
        os.makedirs(self.depth_folder, exist_ok=True)

    def _load_depth_model(self):
        """Load Depth Anything V2 model (small variant for speed)"""
        try:
            import sys
            sys.path.append("Depth-Anything-V2")
            from depth_anything_v2.dpt import DepthAnythingV2
            config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            model = DepthAnythingV2(**config)
            model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vits.pth"))
            return model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Depth model loading failed: {str(e)}")

    def _get_depth_map(self, frame, frame_name):
        """Load precomputed depth map (.npy or .png) if available, otherwise compute and save it"""
        base_name = os.path.splitext(frame_name)[0]  # e.g., "frame_00001"
        depth_npy_path = os.path.join(self.depth_folder, f"{base_name}.npy")
        depth_png_path = os.path.join(self.depth_folder, f"{base_name}.png")

        # Try loading .npy first (raw depth data)
        if os.path.exists(depth_npy_path):
            print(f"Loading precomputed depth map (.npy): {depth_npy_path}")
            return np.load(depth_npy_path)

        # Try loading .png (visualized depth map, assume 0-255 range)
        if os.path.exists(depth_png_path):
            print(f"Loading precomputed depth map (.png): {depth_png_path}")
            depth_img = cv2.imread(depth_png_path, cv2.IMREAD_GRAYSCALE)
            # Convert from 0-255 to 0-10m range (assuming linear scaling from collision detector)
            depth = (depth_img / 255.0) * 10.0
            return depth

        # Compute depth map if neither exists
        print(f"Computing depth map for {frame_name}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(cv2.resize(frame_rgb, (518, 518))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.depth_model(tensor)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=frame.shape[:2],
                                                    mode="bicubic", align_corners=False)
        depth = depth.squeeze().cpu().numpy()
        depth = 10 * (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)  # Normalize to 0-10m

        # Save computed depth map as .npy for future use
        np.save(depth_npy_path, depth)
        print(f"Saved computed depth map to {depth_npy_path}")
        return depth

    def _detect_features(self, gray):
        """Detect SIFT features with emphasis on robust points"""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints if keypoints is not None else [], descriptors

    def _track_features(self, prev_gray, curr_gray, prev_points):
        """Track features using optical flow with validation"""
        if not prev_points:  # Handle empty keypoints list
            return np.array([]), np.array([]), []
        
        pts1 = cv2.KeyPoint_convert(prev_points)
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts1, None,
                                                  winSize=(21, 21), maxLevel=3,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Handle case where tracking fails completely
        if pts2 is None or status is None:
            print("Feature tracking failed, re-detecting features")
            return np.array([]), np.array([]), prev_points
        
        valid = status.squeeze() == 1
        if not np.any(valid):  # No valid points tracked
            return np.array([]), np.array([]), prev_points
        
        return pts1[valid], pts2[valid], np.array(prev_points)[valid].tolist()

    def _estimate_pose(self, pts1, pts2):
        """Estimate camera pose with RANSAC"""
        if len(pts1) < 5 or len(pts2) < 5:  # Minimum points for pose estimation
            return np.eye(3), np.zeros((3, 1))
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t

    def _check_occlusion(self, depth_map):
        """Detect if frame is occluded by a large close object"""
        close_pixels = np.sum(depth_map < self.DEPTH_THRESHOLD)
        coverage = close_pixels / depth_map.size
        return coverage > self.OCCLUSION_COVERAGE

    def _refine_trajectory(self, window_size=5):
        """Bundle adjustment over recent frames (simplified)"""
        if len(self.trajectory) < window_size:
            return
        traj_window = np.array(self.trajectory[-window_size:])
        refined = np.mean(traj_window, axis=0)  # Simple averaging
        self.trajectory[-1] = refined

    def update(self, frame, frame_name):
        """Update camera path based on current frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_map = self._get_depth_map(frame, frame_name)
        self.depth_history.append(np.mean(depth_map))

        # Initialize first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_points, _ = self._detect_features(gray)
            self.last_known_good = {"R": self.R_total.copy(), "t": self.t_total.copy()}
            return

        # Check for occlusion
        if self._check_occlusion(depth_map):
            self.state = "OCCLUDED"
            print("Occlusion detected (e.g., bus in frame), pausing trajectory update")
            return

        # Track features
        pts1, pts2, self.prev_points = self._track_features(self.prev_frame, gray, self.prev_points)

        # Re-detect features if too few or tracking failed
        if len(pts1) < self.MIN_FEATURES:
            self.prev_points, _ = self._detect_features(gray)
            self.state = "STOPPED"
            print("Too few features or tracking failed, stopping and re-detecting")
            return

        # Estimate motion
        R, t = self._estimate_pose(pts1, pts2)

        # Motion check using translation magnitude (tuned for 6 FPS)
        t_magnitude = np.linalg.norm(t)
        if t_magnitude < self.MOTION_THRESHOLD:
            # Secondary depth check only if motion is ambiguous
            depth_change = abs(self.depth_history[-1] - self.depth_history[0]) if len(self.depth_history) > 1 else 0
            if depth_change < 1.0:  # Increased threshold for 6 FPS
                self.state = "STOPPED"
                print(f"Minimal motion detected (t={t_magnitude:.3f}), assuming stopped")
            else:
                self.state = "MOVING"
                self.R_total = R @ self.R_total
                self.t_total += self.R_total @ t
                self.trajectory.append(self.t_total.flatten())
                self.last_known_good = {"R": self.R_total.copy(), "t": self.t_total.copy()}
                self._refine_trajectory()
                print(f"Moving (t={t_magnitude:.3f}), depth change supports motion")
        else:
            self.state = "MOVING"
            self.R_total = R @ self.R_total
            self.t_total += self.R_total @ t
            self.trajectory.append(self.t_total.flatten())
            self.last_known_good = {"R": self.R_total.copy(), "t": self.t_total.copy()}
            self._refine_trajectory()
            print(f"Moving (t={t_magnitude:.3f})")

        self.prev_frame = gray

    def visualize_trajectory(self):
        """Interactive 3D visualization of the camera path"""
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        traj = np.array(self.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='Camera Path', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Camera Trajectory')
        ax.legend()
        plt.show()

    def run(self):
        """Process frames from folder and track path"""
        if not os.path.exists(self.frames_folder):
            raise RuntimeError(f"Frames folder not found: {self.frames_folder}")

        # Get sorted list of frame files
        frame_files = sorted([f for f in os.listdir(self.frames_folder) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            raise RuntimeError(f"No image frames found in: {self.frames_folder}")

        print(f"Processing {len(frame_files)} frames from {self.frames_folder}...")
        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(self.frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_file}")
                continue

            self.update(frame, frame_file)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Processing complete. Visualizing trajectory...")
        self.visualize_trajectory()

if __name__ == "__main__":
    tracker = RobustCameraPathTracker(frames_folder="frames", depth_folder="depth_maps")
    tracker.run()