import pixellib
from pixellib.torchbackend.instance import instanceSegmentation  # Correct import for PointRend (PyTorch)

class SemanticSegmenter:
    def __init__(self, model_path="pointrend_resnet50.pkl"):
        print("🔄 Loading PointRend (ResNet-50) model...")
        self.segmenter = instanceSegmentation()
        try:
            self.segmenter.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")


    def segment(self, frame):
        """
        Perform instance segmentation on the input frame using PointRend.
        """
        if frame is None:
            raise ValueError("❌ Error: Received empty frame!")

        results, output_frame = self.segmenter.segmentFrame(frame, show_bboxes=True, mask_points_values=False)


        if len(results["boxes"]) == 0:
            print("⚠️ No objects detected in this frame!")
        else:
            print(f"✅ Detected {len(results['boxes'])} objects in this frame.")


        print(f"🔍 Segmentation Output Keys: {results.keys()}")  # Debugging print

        # Extract detected objects correctly
        detected_objects = []
        if "boxes" in results and "class_ids" in results:
            for i in range(len(results["boxes"])):
                bbox = results["boxes"][i]  # Bounding box coordinates
                class_id = results["class_ids"][i]  # Object class ID
                confidence = results["scores"][i] if "scores" in results else 1.0  # Confidence score
                
                detected_objects.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "confidence": confidence
                })

        return output_frame, detected_objects
