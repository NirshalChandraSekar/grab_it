from hand_traking import detect_hand_keypoints, sample_points_on_line
from inference_stream import InferenceStream
from segmentation import image_segmentation
from dino_functions import Dinov2
from viz import pca_2d, pca_3d, visualize_rotated_axes

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
    directional_point = keypoints["thumb_tip"]
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
    directional_point_index = dino.pixel_to_idx([directional_point[1], directional_point[0]], demo_image_grid, dino.patch_size)


    '''
    Compute  the cumulative distance map
    '''
    distance_map = None
    for i, point_index in enumerate(point_indices):
        distance = dino.compute_feature_distance(point_index, demo_image_features, inference_color_image_features)
        distance = np.reshape(distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
        distance = cv2.resize(distance, (inference_color_image_tensor.shape[2], 
                                         inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        threshold = np.percentile(distance, 0.005)
        
        distance = distance < threshold

        if distance_map is None:
            distance_map = distance
        else:
            distance_map = np.logical_or(distance_map, distance)


    plt.imshow(inference_color_image)
    plt.imshow(distance_map, alpha=0.5)
    plt.show()

    contact_point_distance = dino.compute_feature_distance(contact_point_index, demo_image_features, inference_color_image_features)
    contact_point_distance = np.reshape(contact_point_distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
    contact_point_distance = cv2.resize(contact_point_distance, (inference_color_image_tensor.shape[2], 
                                         inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(contact_point_distance, 0.005)
    contact_point_distance = contact_point_distance < threshold

    directional_point_distance = dino.compute_feature_distance(directional_point_index, demo_image_features, inference_color_image_features)
    directional_point_distance = np.reshape(directional_point_distance, (inference_color_image_grid[0], inference_color_image_grid[1]))
    directional_point_distance = cv2.resize(directional_point_distance, (inference_color_image_tensor.shape[2], 
                                         inference_color_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(directional_point_distance, 0.005)
    directional_point_distance = directional_point_distance < threshold

    inference_contact_point = np.mean(np.argwhere(contact_point_distance), axis=0)
    inference_directional_point = np.mean(np.argwhere(directional_point_distance), axis=0)
    cv2.circle(inference_color_image, (int(inference_contact_point[1]), int(inference_contact_point[0])), 3, (0, 255, 0), -1)
    cv2.circle(inference_color_image, (int(inference_directional_point[1]), int(inference_directional_point[0])), 3, (0, 255, 0), -1)
    display_image(inference_color_image, "Inference Contact Point Overlay")

   
    '''
    Compute 2D PCA
    '''
    
    important_pixels = np.argwhere(distance_map)
    center = [int(inference_contact_point[1]), int(inference_contact_point[0])]
    pca_2d(important_pixels, center, inference_color_image)
    

    '''
    Compute 3D PCA
    '''
    intrinsics= np.load("resources/" + object_name + "/camera_intrinsic.npy")
    inference_depth_image = np.load("resources/" + object_name + "/inference_depth_image.npy")
    inference_depth_image = inference_depth_image.astype(np.float32)
    inference_depth_image *= 0.001 # D4054
    
    pcd, imp_pcd, contact_point_3d, axes = pca_3d(important_pixels, 
           intrinsics, 
           inference_depth_image, 
           inference_color_image,
           inference_contact_point,
           inference_directional_point)
    
    '''
    Visualize the rotated axes
    '''
    visualize_rotated_axes(pcd, imp_pcd, contact_point_3d, axes)


