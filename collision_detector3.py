import numpy as np
import cv2
import torch
import os
import time
from torchvision.transforms import Compose, ToTensor, Normalize

class EnhancedCollisionDetector:
    def __init__(self, depth_model="vits"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_depth_model(depth_model)
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
            model.load_state_dict(torch.load(f"models_depth_instance/depth_anything_v2_{variant}.pth"))
            return model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"Depth model loading failed: {str(e)}")

    def _dynamic_roi(self, frame_shape):
        h, w = frame_shape[:2]
        return (int(w * 0.3), int(w * 0.7), int(h * 0.3), int(h * 0.7))

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
        
        # Handle negative and positive depths
        depth = depth+np.abs(depth.min())+1  # Convert negative depths to positive
        depth = np.where(depth > 1e-6, 1 / depth, np.inf)  # Convert inverse depth to depth
        depth = depth * 40   #e to meters (adjust based on calibration)
        
        print(f"Depth min: {np.min(depth):.2f}m, max: {np.max(depth):.2f}m")  # Debug
        self._update_performance(start_time)
        return depth
