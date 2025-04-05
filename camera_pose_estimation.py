import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import os
from tqdm import tqdm  # Import tqdm for progress tracking
import time  # For measuring elapsed time

# Camera intrinsic parameters
image_width, image_height = 1920, 1080
focal_length_mm = 24  # Focal length in mm
sensor_width_mm = 36  # Typical full-frame sensor width in mm

# Compute focal length in pixels
fx = (focal_length_mm / sensor_width_mm) * image_width
fy = fx  # Assuming square pixels
cx, cy = image_width / 2, image_height / 2  # Principal point

# Intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Function to load all images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = sorted(os.listdir(folder_path))
    for filename in tqdm(filenames, desc="Loading images", unit="image"):  # Add progress bar
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Feature detection and matching using SIFT
def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Match features using FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched points
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return pts1, pts2

# Camera pose estimation with PnP and RANSAC
def estimate_camera_trajectory(images, K, save_path="trajectory.txt"):
    trajectory = []  # Store camera positions
    R_total = np.eye(3)  # Initial rotation (identity matrix)
    t_total = np.zeros((3, 1))  # Initial translation

    trajectory.append(t_total.flatten())  # Start at origin

    # Initialize tqdm with total number of iterations
    progress_bar = tqdm(total=len(images) - 1, desc="Estimating trajectory", unit="frame")

    for i in range(len(images) - 1):
        # Detect and match features between consecutive frames
        pts1, pts2 = detect_and_match_features(images[i], images[i + 1])

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover pose (rotation and translation)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        # Accumulate rotation and translation
        R_total = R @ R_total
        t_total += R_total @ t

        # Append the camera position
        trajectory.append(t_total.flatten())

        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Save trajectory to file
    np.savetxt(save_path, np.array(trajectory), fmt="%.6f")
    print(f"Trajectory saved to {save_path}")

    return np.array(trajectory)

# Interactive 3D visualization
def visualize_trajectory_3d(trajectory):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    xs, ys, zs = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    trajectory_line, = ax.plot(xs, ys, zs, label='Camera Path', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Interactive 3D Camera Trajectory')
    ax.legend()

    def on_button_press(event):
        # Reset view
        ax.view_init(elev=10, azim=0)
        plt.draw()

    # Add reset view button
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    reset_button = Button(reset_ax, 'Reset View')
    reset_button.on_clicked(on_button_press)

    plt.show()

# Main function
def main():
    folder_path = "frames"  # Folder path for left frames
    print("Loading images...")
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("Not enough images to estimate trajectory.")
        return

    # Estimate camera trajectory and save it
    print("Estimating camera trajectory...")
    start_time = time.time()  # Start timing
    trajectory = estimate_camera_trajectory(images, K)
    end_time = time.time()  # End timing

    # Print total time taken
    total_time = end_time - start_time
    print(f"Trajectory estimation completed in {total_time:.2f} seconds.")

    # Interactive 3D visualization
    print("Visualizing trajectory...")
    visualize_trajectory_3d(trajectory)

if __name__ == "__main__":
    main()