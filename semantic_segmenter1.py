import torch
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize

class AccurateSegmenter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # More accurate class mapping
        self.CLASS_MAP = {
            15: 'person',      # Person
            2:  'car',         # Car
            7:  'truck',       # Truck
            3:  'motorcycle',  # Motorcycle
            6:  'bus',         # Bus
            1:  'road-hazard'  # Road obstacles
        }
        
        # Conservative distance thresholds (meters)
        self.MIN_DISTANCES = {
            'person': 1.0,
            'car': 1.0,
            'truck': 1.0,
            'motorcycle': 1.0,
            'bus': 1.0,
            'road-hazard': 1.0
        }

    def segment(self, image):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output = output.argmax(0).cpu().numpy()
        
        return output

    def get_masks(self, segmented):
        masks = {}
        for class_id, class_name in self.CLASS_MAP.items():
            if class_id in segmented:
                masks[class_name] = (
                    segmented == class_id,
                    self.MIN_DISTANCES[class_name]
                )
        return masks

    def visualize(self, image, segmented):
        vis = image.copy()
        overlay = np.zeros_like(image)
        
        for class_id, class_name in self.CLASS_MAP.items():
            if class_id in segmented:
                color = self._get_color(class_name)
                overlay[segmented == class_id] = color
        
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        return vis

    def _get_color(self, class_name):
        return {
            'person': [255, 0, 0],
            'car': [0, 0, 255],
            'truck': [0, 100, 255],
            'motorcycle': [0, 255, 255],
            'bus': [255, 255, 0],
            'road-hazard': [255, 0, 255]
        }.get(class_name, [0, 0, 0])