import torch
import numpy as np
import cv2
import time
from typing import Dict, Tuple
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

class InstanceSegmenter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
            self.model.to(self.device).eval()
            self.transform = weights.transforms()
        except ImportError:
            from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.7)
            self.model.to(self.device).eval()
            self.transform = weights.transforms()
        
        self.target_classes = {
            1: 'person',
            2: 'bicycle',
            3: 'car', 
            4: 'motorcycle',
            5: 'airplane',
            6: 'bus',
            7: 'train',
            8: 'truck', 
            13: 'bench',
            17: 'cat',
            18: 'dog'
        }
        
        self.frame_count = 0
        self.avg_process_time = 0

    def detect(self, frame: np.ndarray) -> Tuple[Dict, float]:
        """Detect objects in the frame, return class, box, and score"""
        start_time = time.time()
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        objects = self._process_predictions(predictions[0], frame.shape[:2])
        
        process_time = time.time() - start_time
        self.frame_count += 1
        self.avg_process_time = 0.9 * self.avg_process_time + 0.1 * process_time
        
        return objects, process_time

    def _process_predictions(self, prediction: Dict, img_shape: Tuple[int, int]) -> Dict:
        """Process model predictions into objects with class, box, score"""
        objects = {}
        
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy()
        
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            if score < 0.7 or label not in self.target_classes:
                continue
                
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            area = np.sum(binary_mask)
            if area < 1000:
                continue
                
            class_name = self.target_classes[label.item()]
            instance_id = f"{class_name}_{i}"
            objects[instance_id] = {
                'class_name': class_name,
                'score': score.item(),
                'box': box.astype(int),
                'mask': binary_mask  # Keep mask for visualization
            }
            
        return objects

    def visualize(self, frame: np.ndarray, objects: Dict, depth_map: np.ndarray = None) -> np.ndarray:
        """Create visualization with bounding boxes"""
        vis = frame.copy()
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for obj_id, obj in objects.items():
            class_name = obj['class_name']
            distance = obj.get('distance', float('inf'))
            
            # Color based on distance
            if distance < 6.5:
                color = (0, 0, 255)  # Red for STOP
            elif distance < 10.0:
                color = (0, 165, 255)  # Orange for WARNING
            else:
                color = (0, 255, 0)  # Green for CLEAR
                
            # Draw mask on overlay
            overlay[obj['mask'] > 0] = color
            
            # Draw bounding box
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Add label with distance
            label = f"{class_name}: {distance:.1f}m"
            cv2.putText(vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        return vis