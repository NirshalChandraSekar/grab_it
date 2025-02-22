from hand_traking import detect_hand_keypoints, sample_points_on_line
from inference_stream import InferenceStream
from segmentation import image_segmentation
from dino_functions import Dinov2
from viz import pca_2d, pca_3d, get_gt, visualize_rotated_axes

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
    object_name = "pouch"

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
    contact_point_2d = {}
    directional_point_2d = {}

    for key in keypoints:
        contact_point_2d[key] = [int(abs(keypoints[key]["thumb_tip"][0]+keypoints[key]["thumb_ip"][0])/2),
                              int(abs(keypoints[key]["thumb_tip"][1]+keypoints[key]["thumb_ip"][1])/2)]
        directional_point_2d[key] = keypoints[key]["thumb_tip"]
        cv2.circle(keypoint_overlay, keypoints[key]["thumb_tip"], 3, (0, 0, 255), -1)
        cv2.circle(keypoint_overlay, keypoints[key]["thumb_ip"], 3, (0, 0, 255),  -1)
        cv2.circle(keypoint_overlay, contact_point_2d[key], 3, (0, 255, 0), -1)

    display_image(keypoint_overlay, "Hand Keypoints Overlay")


    checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentor = image_segmentation(checkpoint=checkpoint, config=config, device=device)
    mask = segmentor.segment_image(demo_image)
    mask = mask.transpose(1, 2, 0)
    viewable_mask = mask.astype(np.uint8) * 255
    display_image(viewable_mask, "Segmented Mask")

    sampled_points_2d = {}
    for key in keypoints:
        sampled_points_2d[key] = sample_points_on_line(keypoints[key]["thumb_tip"], keypoints[key]["thumb_ip"], mask)
        for point in sampled_points_2d[key]:
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



    point_indices = {}
    contact_point_index = {}
    directional_point_index = {}

    for key in sampled_points_2d:
        point_indices[key] = []
        for point in sampled_points_2d[key]:
            col, row = point
            point_indices[key].append(dino.pixel_to_idx([row, col], demo_image_grid, dino.patch_size))
        contact_point_index[key] = dino.pixel_to_idx([contact_point_2d[key][1], contact_point_2d[key][0]], demo_image_grid, dino.patch_size)
        directional_point_index[key] = dino.pixel_to_idx([directional_point_2d[key][1], directional_point_2d[key][0]], demo_image_grid, dino.patch_size)


    '''
    Compute  the cumulative distance map
    '''
    distance_map = {}
    for key in point_indices:
        distance_map[key] = None
        for i, point_index in enumerate(point_indices[key]):
            distance = dino.compute_feature_distance(point_index, demo_image_features, inference_color_image_features)
            distance = np.reshape(distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
            distance = cv2.resize(distance, (inference_color_image_tensor.shape[2], 
                                            inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
            threshold = np.percentile(distance, 0.005)
            distance = distance < threshold

            if distance_map[key] is None:
                distance_map[key] = distance
            else:
                distance_map[key] = np.logical_or(distance_map[key], distance)


    plt.imshow(inference_color_image)
    for key in distance_map:
        plt.imshow(distance_map[key], alpha=0.5)
    plt.show()

    contact_point_distance = {}
    directional_point_distance = {}
    inference_contact_point = {}
    inference_directional_point = {}
    for key in point_indices:
        contact_point_distance[key] = dino.compute_feature_distance(contact_point_index[key], demo_image_features, inference_color_image_features)
        contact_point_distance[key] = np.reshape(contact_point_distance[key], (inference_color_image_grid[0], inference_color_image_grid[1]))
        contact_point_distance[key] = cv2.resize(contact_point_distance[key], (inference_color_image_tensor.shape[2],
                                             inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        threshold = np.percentile(contact_point_distance[key], 0.005)
        contact_point_distance[key] = contact_point_distance[key] < threshold   

        directional_point_distance[key] = dino.compute_feature_distance(directional_point_index[key], demo_image_features, inference_color_image_features)
        directional_point_distance[key] = np.reshape(directional_point_distance[key], (inference_color_image_grid[0], inference_color_image_grid[1]))
        directional_point_distance[key] = cv2.resize(directional_point_distance[key], (inference_color_image_tensor.shape[2],
                                                inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        threshold = np.percentile(directional_point_distance[key], 0.005)
        directional_point_distance[key] = directional_point_distance[key] < threshold

        inference_contact_point[key] = np.mean(np.argwhere(contact_point_distance[key]), axis=0)
        inference_directional_point[key] = np.mean(np.argwhere(directional_point_distance[key]), axis=0)

        cv2.circle(inference_color_image, (int(inference_contact_point[key][1]), int(inference_contact_point[key][0])), 3, (0, 255, 0), -1)
        cv2.circle(inference_color_image, (int(inference_directional_point[key][1]), int(inference_directional_point[key][0])), 3, (0, 255, 0), -1)

    display_image(inference_color_image, "Inference Contact Point Overlay")


   
    '''
    Compute 2D PCA
    '''
    important_pixels = {}
    center = {}
    for key in distance_map:
        important_pixels[key] = np.argwhere(distance_map[key])
        center[key] = [int(inference_contact_point[key][1]), int(inference_contact_point[key][0])]

    # pca_2d(important_pixels, center, inference_color_image)

    '''
    Get Ground Truth Grasp Axes from the user
    '''
    intrinsics= np.load("resources/" + object_name + "/camera_intrinsic.npy")
    inference_depth_image = np.load("resources/" + object_name + "/inference_depth_image.npy")
    inference_depth_image = inference_depth_image.astype(np.float32)
    inference_depth_image *= 0.001 # D4054
    # inference_depth_image *= 0.00025

    gt_grasp_axes = {}
    for key in important_pixels:
        gt_grasp_axes[key] = get_gt(inference_color_image, inference_depth_image, intrinsics)
    print("GT Grasp Axes: ", gt_grasp_axes)


    

    '''
    Compute 3D PCA
    '''
    
    
    grasp_axes = pca_3d(important_pixels, 
           intrinsics, 
           inference_depth_image, 
           inference_color_image,
           inference_contact_point,
           inference_directional_point)
    
    print("Grasp Axes: ", grasp_axes)
    
    '''
    Visualize the rotated axes
    '''
    # visualize_rotated_axes(pcd, imp_pcd, contact_point_3d, axes)


