import numpy as np
import cv2
from collections import deque
import torch
import os
import time
from torchvision.transforms import Compose, ToTensor, Normalize

class EnhancedCollisionDetector:
    def __init__(self, depth_model="vits", warning_dist=5.0, stop_dist=3.0, temporal_window=3):
        self.warning_distance = warning_dist
        self.stop_distance = stop_dist
        self.temporal_window = temporal_window
        self.stop_history = deque(maxlen=temporal_window)
        self.warning_history = deque(maxlen=temporal_window)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_depth_model(depth_model)
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.min_coverage = 0.05
        self.critical_coverage = 0.15
        self.confidence_threshold = 0.3
        self.frame_count = 0
        self.avg_process_time = 0

    def _load_depth_model(self, variant="vits"):
        try:
            import sys
            sys.path.append("Depth-Anything-V2")
            from depth_anything_v2.dpt import DepthAnythingV2
            
            configs = {
                "vits": {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
            }
            
            model = DepthAnythingV2(**configs[variant])
            model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{variant}.pth"))
            return model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Depth model loading failed: {str(e)}")

    def _dynamic_roi(self, frame_shape):
        h, w = frame_shape[:2]
        return (int(w * 0.2), int(w * 0.8), int(h * 0.3), int(h * 0.7))

    def _get_object_coverage(self, depth_map, roi_coords):
        x_s, x_e, y_s, y_e = roi_coords
        roi = depth_map[y_s:y_e, x_s:x_e]
        valid_pixels = (roi > 1.0) & (roi < self.warning_distance)
        return np.sum(valid_pixels) / roi.size

    def _update_performance(self, start_time):
        """Track average processing time"""
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.avg_process_time = 0.9 * self.avg_process_time + 0.1 * elapsed

    def get_depth_map(self, frame):
        start_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(cv2.resize(frame_rgb, (518, 518))).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()
        
        # CALIBRATION CRITICAL - Add your camera's real-world scale factor
        # Example: If 1.0 in depth map = 15 real-world meters
        depth = depth * 15.0  # Change 15.0 to your calibration constant
        
        self._update_performance(start_time)
        return depth


    def analyze_depth(self, depth_map):
        """Analyze depth map for obstacles"""
        if depth_map is None:
            return float('inf'), 0.0
        
        roi_coords = self._dynamic_roi(depth_map.shape)
        coverage = self._get_object_coverage(depth_map, roi_coords)
        
        if coverage > self.min_coverage:
            x_s, x_e, y_s, y_e = roi_coords
            roi = depth_map[y_s:y_e, x_s:x_e]
            valid_depths = roi[(roi > 1.0) & (roi < self.warning_distance)]
            
            if len(valid_depths) > 0:
                min_depth = np.min(valid_depths)
                confidence = min(1.0, coverage * 2)
                return min_depth, confidence
        
        return float('inf'), 1.0

    def check_collision(self, frame, frame_idx=None, semantic_data=None):
        depth_map = self.get_depth_map(frame)
        
        # DEBUG: Print actual depth values
        print(f"Max depth: {np.max(depth_map):.1f}m, Min depth: {np.min(depth_map[depth_map > 0]):.1f}m")
        
        if semantic_data:
            roi_coords = self._dynamic_roi(depth_map.shape)
            x_s, x_e, y_s, y_e = roi_coords
            
            for class_name, (mask, min_dist) in semantic_data.items():
                roi_mask = mask[y_s:y_e, x_s:x_e]
                roi_depth = depth_map[y_s:y_e, x_s:x_e]
                
                valid_mask = roi_mask & (roi_depth > 0.5)  # Only consider valid depths
                valid_depths = roi_depth[valid_mask]
                
                if valid_depths.size > 0:
                    min_depth = np.min(valid_depths)
                    print(f"Detected {class_name} at {min_depth:.1f}m")  # Debug output
                    
                    if min_depth < min_dist:
                        return f"STOP - {class_name} ({min_depth:.1f}m)", min_depth, depth_map
            
        
        # Original depth-based fallback
        min_depth, confidence = self.analyze_depth(depth_map)
        coverage = self._get_object_coverage(depth_map, self._dynamic_roi(depth_map.shape))
        
        if coverage > self.critical_coverage and confidence > 0.7:
            return "WARNING", min_depth, depth_map
        
        return "CLEAR", min_depth, depth_map

    def visualize(self, frame, depth_map, status, distance):
        vis = frame.copy()
        x_s, x_e, y_s, y_e = self._dynamic_roi(frame.shape)
        
        color = (0, 255, 0) if status == "CLEAR" else \
               (0, 165, 255) if status == "WARNING" else \
               (0, 0, 255)
        
        cv2.rectangle(vis, (x_s, y_s), (x_e, y_e), color, 3)
        
        coverage = self._get_object_coverage(depth_map, (x_s, x_e, y_s, y_e))
        cv2.putText(vis, f"Coverage: {coverage*100:.1f}%", (x_s, y_s-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        status_text = f"{status} | {distance:.1f}m | {self.avg_process_time*1000:.1f}ms"
        cv2.putText(vis, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis

    def save_depth_map(self, depth_map, frame_idx, output_dir="depth_maps"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/depth_{frame_idx:05d}.npy", depth_map)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"{output_dir}/depth_{frame_idx:05d}.jpg", 
                   cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET))

    def load_depth_map(self, frame_idx, output_dir="depth_maps"):
        npy_path = f"{output_dir}/depth_{frame_idx:05d}.npy"
        return np.load(npy_path) if os.path.exists(npy_path) else None
