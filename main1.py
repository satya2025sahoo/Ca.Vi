import os
import cv2
import argparse
import numpy as np
from semantic_segmenter import SemanticSegmenter
from object_tracker import ObjectTracker
from object_detector import ObjectDetector
from navigator import Navigator
from map_generator import MapGenerator
from collision_detector import CollisionDetector

def extract_frames(video_path, frame_dir, target_fps=6):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    if os.listdir(frame_dir):
        print("✅ Frames already exist. Skipping extraction.")
        return
    print("📽 Extracting frames from video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Unable to open video file!")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(original_fps / target_fps))
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frame_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"✅ Frame extraction complete! {saved_count} frames saved.")

def process_frames(frame_dir, output_dir, map_output_dir, depth_output_dir):
    if not os.path.exists(frame_dir) or not os.listdir(frame_dir):
        print("❌ Error: No frames found in the 'frames' directory. Exiting process.")
        return

    # Ensure output directories exist
    for dir_path in [output_dir, map_output_dir, depth_output_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Initialize components
    segmenter = SemanticSegmenter()
    tracker = ObjectTracker(max_age=5, min_hits=3, iou_threshold=0.3)
    detector = ObjectDetector()
    navigator = Navigator()
    map_gen = MapGenerator(map_size=(800, 800))

    # Determine depth map shape from the first frame
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print("❌ No frames found in the frame directory!")
        return
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"❌ Failed to load first frame: {first_frame_path}")
        return
    frame_height, frame_width = first_frame.shape[:2]
    depth_map_shape = (frame_height, frame_width)

    # Initialize CollisionDetector once (outside the loop for efficiency)
    collision_detector = CollisionDetector(depth_map_shape=depth_map_shape, depth_output_dir=depth_output_dir)

    # Check existing depth maps in depth_output_dir
    existing_depth_maps = sorted([f for f in os.listdir(depth_output_dir) if f.endswith('.npy') or f.endswith('.jpg')])
    depth_map_indices = {int(f.split('_')[1].split('.')[0]): f for f in existing_depth_maps}
    print(f"✅ Found {len(depth_map_indices)} existing depth maps in {depth_output_dir}")

    position_history = [(0, "clear")]
    MIN_BBOX_AREA = 400

    for frame_index, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"⚠️ Warning: Unable to read {frame_name}, skipping.")
            continue

        # Step 1: Object Detection with ObjectDetector (YOLOv8)
        detected_objects = detector.detect(frame)
        detected_objects = [obj for obj in detected_objects if (obj[2][2] - obj[2][0]) * (obj[2][3] - obj[2][1]) > MIN_BBOX_AREA]

        # Prepare detections for ObjectTracker (format: [x1, y1, x2, y2, score])
        detections = []
        for obj in detected_objects:
            _, score, bbox = obj
            x1, y1, x2, y2 = bbox
            detections.append([x1, y1, x2, y2, score])

        # Step 2: Segmentation with SemanticSegmenter (PointRend)
        segmented_frame, _ = segmenter.segment(frame)
        segmentation_masks = {}
        for idx, det in enumerate(detected_objects):
            _, _, bbox = det
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)  # Placeholder
            mask[10:-10, 10:-10] = 1  # Dummy mask
            full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask
            segmentation_masks[idx] = full_mask

        # Step 3: Object Tracking with ObjectTracker
        if detections:
            tracks, tracked_frame = tracker.update(detections, segmentation_masks, frame)
        else:
            print("⚠️ No objects detected, using last known tracked objects.")
            tracks, tracked_frame = tracker.update(np.empty((0, 5)), {}, frame)

        # Step 4: Depth Map and Collision Detection
        # Check if a depth map exists for this frame_index
        depth_map = collision_detector.load_depth_map(frame_index)
        if depth_map is None:
            print(f"Generating depth map for frame {frame_index} using original frame...")
            # Use the original frame for depth estimation, not the segmented one
            depth_map = collision_detector.generate_depth_map(frame)
            collision_detector.save_depth_map(depth_map, frame_index)
        else:
            print(f"✅ Using existing depth map for frame {frame_index}")

        state, distance = collision_detector.check_collision(depth_map)
        stop_needed = (state == "STOP")

        # Step 5: Navigation Decision
        command = "STOP" if stop_needed else "MOVE"
        navigator.speak(command)
        direction = "stop" if stop_needed else "straight"
        position_history.append((frame_index, direction))

        # Step 6: Update Map with MapGenerator
        map_image = map_gen.update_map(position_history, depth_map, frame)
        if map_image is None:
            print("❌ Map image not generated!")
            map_image = np.zeros((800, 800, 3), dtype=np.uint8)

        # Step 7: Visualize and Save Depth Map
        depth_vis = collision_detector.visualize_depth(depth_map)
        if depth_vis is None:
            print("❌ Depth visualization not generated!")
            depth_vis = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        depth_map_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))

        # Annotate depth map for display
        move_stop_text = "STOP - COLLISION RISK" if stop_needed else "MOVE - SAFE PATH"
        move_stop_color = (0, 0, 255) if stop_needed else (0, 255, 0)
        cv2.putText(depth_map_resized, move_stop_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, move_stop_color, 2)
        cv2.putText(depth_map_resized, f"Direction: {direction.upper()}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Step 8: Combine RGB frame and depth map
        final_output = np.hstack((tracked_frame, depth_map_resized))

        # Step 9: Save Outputs
        cv2.imwrite(os.path.join(output_dir, frame_name), final_output)
        cv2.imwrite(os.path.join(map_output_dir, f"map_{frame_index:04d}.jpg"), map_image)
        cv2.imwrite(os.path.join(depth_output_dir, f"depth_{frame_index:04d}.jpg"), depth_map_resized)

        # Step 10: Display
        cv2.imshow("Annotated Image", final_output)
        cv2.imshow("Generated Navigation Map", map_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"✅ Frame {frame_name} processed:")
        print(f"   - Objects detected: {len(detected_objects)}")
        print(f"   - Movement: {direction.upper()}")
        print(f"   - Collision Risk: {'STOP' if stop_needed else 'MOVE'}")
        print(f"   - Output saved at: {output_dir}/{frame_name}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with segmentation, tracking, detection, navigation, and collision detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()
    
    frame_dir = "frames"
    output_dir = "output"
    map_output_dir = "map_output"
    depth_output_dir = "depth_output"
    
    extract_frames(args.video, frame_dir, target_fps=6)
    process_frames(frame_dir, output_dir, map_output_dir, depth_output_dir)
    
    print("🎉 All frames processed successfully!")
