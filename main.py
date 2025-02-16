from hand_traking import detect_hand_keypoints, sample_points_on_line
from inference_stream import InferenceStream
from segmentation import image_segmentation
from dino_functions import Dinov2

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
import cv2

def display_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    object_name = "dino"

    '''
    Read the demonstration images. Ideally we will have to give the video as input.
    But for simplicity we are using images for now.
    Later we will use the video as input.
    '''
    demo_image = cv2.imread("resources/" + object_name + "/demo.jpg")
    demo_image_hand = cv2.imread("resources/" + object_name + "/demo_hand.jpg")

    demo_image = cv2.resize(demo_image, (demo_image.shape[1]//5,demo_image.shape[0]//5))
    demo_image_hand = cv2.resize(demo_image_hand, (demo_image_hand.shape[1]//5,demo_image_hand.shape[0]//5))



    '''
    Run Mediapipe on the hand image to get the keypoints of thumb tip and thumb ip
    '''
    keypoints = detect_hand_keypoints(demo_image_hand)
    keypoint_overlay = demo_image.copy()
    contact_point = [int(abs(keypoints["thumb_tip"][0]+keypoints["thumb_ip"][0])/2), 
                    int(abs(keypoints["thumb_tip"][1]+keypoints["thumb_ip"][1])/2)]
    cv2.circle(keypoint_overlay, keypoints["thumb_tip"], 3, (0, 0, 255), -1)
    cv2.circle(keypoint_overlay, keypoints["thumb_ip"], 3, (0, 0, 255),  -1)
    cv2.circle(keypoint_overlay, contact_point, 3, (0, 255, 0), -1)
    display_image(keypoint_overlay, "Hand Keypoints Overlay")


    checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentor = image_segmentation(checkpoint=checkpoint, config=config, device=device)
    mask = segmentor.segment_image(demo_image)
    mask = mask.transpose(1, 2, 0)
    viewable_mask = mask.astype(np.uint8) * 255
    display_image(viewable_mask, "Segmented Mask")


    sampled_points = sample_points_on_line(keypoints["thumb_tip"], keypoints["thumb_ip"], mask)
    for point in sampled_points:
        cv2.circle(keypoint_overlay, point, 3, (0, 255, 0), -1)

    display_image(keypoint_overlay, "Sampled Points Overlay")



    '''
    Record Inference using the realsense camera
    '''
    check = input("Do you want to record the inference? (1 for yes, 0 for no): ")
    if check == "1":
        _ = input("Verify if camera is connected and press enter to continue")
        stream = InferenceStream()
        color_image, depth_image, intrinsic = stream.get_frame()
        display_image(color_image, "Inference Image")
        display_image(depth_image, "Inference Depth Image")
        np.save("resources/" + object_name + "/inference_color_image.npy", color_image)
        np.save("resources/" + object_name + "/inference_depth_image.npy", depth_image)
        np.save("resources/" + object_name + "/camera_intrinsic.npy", intrinsic)
    
    inference_color_image = np.load("resources/" + object_name + "/inference_color_image.npy")



    '''
    Run the dino functions for feature extraction
    '''
    demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
    inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)
    
    dino = Dinov2()
    print("dino initialized")
    
    demo_image_tensor, demo_image_grid = dino.prepare_image(demo_image)
    inference_color_image_tensor, inference_color_image_grid = dino.prepare_image(inference_color_image)

    demo_image_features = dino.extract_features(demo_image_tensor)
    inference_color_image_features = dino.extract_features(inference_color_image_tensor)

    print("Demo Image Features Shape: ", demo_image_features.shape)
    print("Inference Image Features Shape: ", inference_color_image_features.shape)


    point_indices = []
    for point in sampled_points:
        col, row = point
        point_indices.append(dino.pixel_to_idx([row, col], demo_image_grid, dino.patch_size))
    
    contact_point_index = dino.pixel_to_idx([contact_point[1], contact_point[0]], demo_image_grid, dino.patch_size)



    '''
    Compute  the cumulative distance map
    '''
    distance_map = None
    for i, point_index in enumerate(point_indices):
        distance = dino.compute_feature_distance(point_index, demo_image_features, inference_color_image_features)
        distance = np.reshape(distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
        distance = cv2.resize(distance, (inference_color_image_tensor.shape[2], 
                                         inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        if distance_map is None:
            distance_map = distance
        else:
            distance_map += distance

    plt.imshow(inference_color_image)
    plt.imshow(distance_map, alpha=0.5)
    plt.colorbar()
    plt.show()

    contact_point_distance = dino.compute_feature_distance(contact_point_index, demo_image_features, inference_color_image_features)
    contact_point_distance = np.reshape(contact_point_distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
    contact_point_distance = cv2.resize(contact_point_distance, (inference_color_image_tensor.shape[2], 
                                         inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
    
    inference_contact_point = np.unravel_index(np.argmin(contact_point_distance, axis=None), contact_point_distance.shape)


    '''
    Compute 2D PCA
    '''
    threshold = np.percentile(distance_map, 0.05)
    important_pixels = np.argwhere(distance_map < threshold)

    pca = PCA(n_components=2)
    pca.fit(important_pixels)

    center = [inference_contact_point[1], inference_contact_point[0]]
    direction = pca.components_[0]

    length = 30
    p1 = (int(center[0] - length * direction[0]), int(center[1] - length * direction[1]))
    p2 = (int(center[0] + length * direction[0]), int(center[1] + length * direction[1]))

    pca_image = inference_color_image.copy()
    pca_image = cv2.cvtColor(pca_image, cv2.COLOR_RGB2BGR)
    cv2.line(pca_image, p1, p2, (0, 255, 0), 2)
    cv2.circle(pca_image, center, 3, (0, 0, 255), -1)

    display_image(pca_image, "PCA Line Overlay")


    '''
    Compute 3D PCA
    '''
    cx, cy, fx, fy = np.load("resources/" + object_name + "/camera_intrinsic.npy")
    inference_depth_image = np.load("resources/" + object_name + "/inference_depth_image.npy")
    inference_depth_image = inference_depth_image.astype(np.float32)
    inference_depth_image *= 0.001 # D4054
    # inference_depth_image *= 0.00025 # L515

    intrinsics = o3d.camera.PinholeCameraIntrinsic(inference_color_image.shape[1], inference_color_image.shape[0], fx, fy, cx, cy)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(inference_color_image), 
                            o3d.geometry.Image(inference_depth_image), 
                            depth_scale=1.0, 
                            depth_trunc=3, 
                            convert_rgb_to_intensity=False
                            )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    v_coords = important_pixels[:, 0]
    u_coords = important_pixels[:, 1]

    z_values = inference_depth_image[v_coords, u_coords]
    
    x_values = (u_coords - cx) * z_values / fx
    y_values = (v_coords - cy) * z_values / fy

    important_points_3d = np.column_stack([x_values, y_values, z_values])

    inference_contact_point_3d = [(inference_contact_point[1] - cx) * inference_depth_image[inference_contact_point[0], inference_contact_point[1]] / fx,
                                  (inference_contact_point[0] - cy) * inference_depth_image[inference_contact_point[0], inference_contact_point[1]] / fy,
                                  inference_depth_image[inference_contact_point[0], inference_contact_point[1]]]
    
    pcd_important = o3d.geometry.PointCloud()
    pcd_important.points = o3d.utility.Vector3dVector(important_points_3d)
    pcd_important.paint_uniform_color([1, 0, 0])

    pca = PCA(n_components=3)
    pca.fit(important_points_3d)

    center = inference_contact_point_3d
    axes = pca.components_

    scale = 0.5
    axis_points = np.array([
        center, center + scale * axes[0],  # Principal axis 1
        center, center + scale * axes[1],  # Principal axis 2
        center, center + scale * axes[2],  # Principal axis 3
    ])

    lines = [[0, 1], [2, 3], [4, 5]]  # Each pair defines a line
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB: Red, Green, Blue
    axis_lines = o3d.geometry.LineSet()
    axis_lines.points = o3d.utility.Vector3dVector(axis_points)
    axis_lines.lines = o3d.utility.Vector2iVector(lines)
    axis_lines.colors = o3d.utility.Vector3dVector(colors)  # Set colors

    o3d.visualization.draw_geometries([pcd, pcd_important, axis_lines])

    