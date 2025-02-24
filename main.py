from hand_traking import detect_hand_keypoints, sample_points_on_line
from inference_stream import InferenceStream, InferenceMultiCamera
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

    # display_image(keypoint_overlay, "Hand Keypoints Overlay")


    checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentor = image_segmentation(checkpoint=checkpoint, config=config, device=device)
    mask = segmentor.segment_image(demo_image)
    mask = mask.transpose(1, 2, 0)
    viewable_mask = mask.astype(np.uint8) * 255
    # display_image(viewable_mask, "Segmented Mask")

    sampled_points_2d = {}
    for key in keypoints:
        sampled_points_2d[key] = sample_points_on_line(keypoints[key]["thumb_tip"], keypoints[key]["thumb_ip"], mask)
        for point in sampled_points_2d[key]:
            cv2.circle(keypoint_overlay, point, 3, (0, 255, 0), -1)

    # display_image(keypoint_overlay, "Sampled Points Overlay")



    '''
    Record Inference using the realsense camera
    '''
    check = input("Do you want to record the inference? (1 for yes, 0 for no): ")
    if check == "1":
        _ = input("Verify if camera is connected and press enter to continue")
        # stream = InferenceStream()
        # color_image, depth_image, intrinsic = stream.get_frame()
        # display_image(color_image, "Inference Image")
        # display_image(depth_image, "Inference Depth Image")
        # np.save("resources/" + object_name + "/inference_color_image.npy", color_image)
        # np.save("resources/" + object_name + "/inference_depth_image.npy", depth_image)
        # np.save("resources/" + object_name + "/camera_intrinsic.npy", intrinsic)
        
        stream = InferenceMultiCamera()
        frames_dict = stream.get_frames()
        for key in frames_dict:
            color_image = frames_dict[key]["color"]
            depth_image = frames_dict[key]["depth"]
            intrinsic = [frames_dict[key]["intrinsics"]['ppx'],
                         frames_dict[key]["intrinsics"]['ppy'],
                         frames_dict[key]["intrinsics"]['fx'],
                         frames_dict[key]["intrinsics"]['fy']]
            np.save("resources/" + object_name + "/inference_color_image_" + key + ".npy", color_image)
            np.save("resources/" + object_name + "/inference_depth_image_" + key + ".npy", depth_image)
            np.save("resources/" + object_name + "/camera_intrinsic_" + key + ".npy", intrinsic)
    



    '''
    Run the dino functions for feature extraction
    '''
    dino = Dinov2()
    demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
    
    demo_image_tensor, demo_image_grid = dino.prepare_image(demo_image)
    demo_image_features = dino.extract_features(demo_image_tensor)

    indices_sampled_points_2d = {}
    demo_contact_point_index = {}
    demo_directional_point_index = {}
    for key in sampled_points_2d:
        indices_sampled_points_2d[key] = []
        for point in sampled_points_2d[key]:
            col, row = point
            indices_sampled_points_2d[key].append(dino.pixel_to_idx([row, col], demo_image_grid, dino.patch_size))
        demo_contact_point_index[key] = dino.pixel_to_idx([contact_point_2d[key][1], contact_point_2d[key][0]], demo_image_grid, dino.patch_size)
        demo_directional_point_index[key] = dino.pixel_to_idx([directional_point_2d[key][1], directional_point_2d[key][0]], demo_image_grid, dino.patch_size)

    
    camera_serials = [130322273305, 128422270081, 127122270512]
    best_serial = None
    min_distance = float('inf')
    best_inference_contact_point = None
    best_inference_directional_point = None
    best_distance_map = None

    for serial in camera_serials:
        inference_color_image = np.load("resources/" + object_name + "/inference_color_image_" + str(serial) + ".npy")
        inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)

        inference_image_tensor, inference_image_grid = dino.prepare_image(inference_color_image)
        inference_image_features = dino.extract_features(inference_image_tensor)

        total_distance = 0
        dist_map = {}
        contact_pt = {}
        directional_pt = {}

        for key in indices_sampled_points_2d:
            dist_map[key] = None
            for i, point_index in enumerate(indices_sampled_points_2d[key]):
                distance = dino.compute_feature_distance(point_index,
                                                         demo_image_features,
                                                         inference_image_features)
                distance = np.reshape(distance, (inference_image_grid[0], inference_image_grid[1]))
                distance = cv2.resize(distance, (inference_image_tensor.shape[2],
                                                 inference_image_tensor.shape[1]),
                                                 interpolation=cv2.INTER_CUBIC)
                threshold = np.percentile(distance, 0.005)
                distance_mask = distance < threshold
                distance_mask = distance_mask.astype(np.uint8)
                distance *= distance_mask

                if dist_map[key] is None:
                    dist_map[key] = distance
                else:
                    dist_map[key] = np.logical_or(dist_map[key], distance>0)

                total_distance += np.sum(distance)

            contact_pt_distance = dino.compute_feature_distance(demo_contact_point_index[key],
                                                                demo_image_features,
                                                                inference_image_features)
            contact_pt_distance = np.reshape(contact_pt_distance, (inference_image_grid[0], inference_image_grid[1]))
            contact_pt_distance = cv2.resize(contact_pt_distance, (inference_image_tensor.shape[2],
                                                                 inference_image_tensor.shape[1]),
                                                                 interpolation=cv2.INTER_CUBIC)
            contact_pt_threshold = np.percentile(contact_pt_distance, 0.005)
            contact_pt_distance_mask = contact_pt_distance < contact_pt_threshold
            contact_pt_distance_mask = contact_pt_distance_mask.astype(np.uint8)
            contact_pt_distance *= contact_pt_distance_mask
            contact_pt[key] = np.mean(np.argwhere(contact_pt_distance_mask>0), axis=0)

            directional_pt_distance = dino.compute_feature_distance(demo_directional_point_index[key],
                                                                   demo_image_features,
                                                                   inference_image_features)
            directional_pt_distance = np.reshape(directional_pt_distance, (inference_image_grid[0], inference_image_grid[1]))
            directional_pt_distance = cv2.resize(directional_pt_distance, (inference_image_tensor.shape[2],
                                                                       inference_image_tensor.shape[1]),
                                                                       interpolation=cv2.INTER_CUBIC)
            directional_pt_threshold = np.percentile(directional_pt_distance, 0.005)
            directional_pt_distance_mask = directional_pt_distance < directional_pt_threshold
            directional_pt_distance_mask = directional_pt_distance_mask.astype(np.uint8)
            directional_pt_distance *= directional_pt_distance_mask 
            directional_pt[key] = np.mean(np.argwhere(directional_pt_distance_mask>0), axis=0)

            total_distance += np.sum(contact_pt_distance)
            total_distance += np.sum(directional_pt_distance)

        if total_distance < min_distance:
            min_distance = total_distance
            best_serial = serial
            best_distance_map = dist_map
            best_inference_contact_point = contact_pt
            best_inference_directional_point = directional_pt

    print("Best Serial: ", best_serial)
    inference_color_image = np.load("resources/" + object_name + "/inference_color_image_" + str(best_serial) + ".npy")
    inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)

    distance_map = best_distance_map
    inference_contact_point = best_inference_contact_point
    inference_directional_point = best_inference_directional_point

                




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

    intrinsics = np.load("resources/" + object_name + "/camera_intrinsic_" + str(best_serial) + ".npy")
    inference_depth_image = np.load("resources/" + object_name + "/inference_depth_image_" + str(best_serial) + ".npy")
    inference_color_image = np.load("resources/" + object_name + "/inference_color_image_" + str(best_serial) + ".npy")
    inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)
    inference_depth_image = inference_depth_image.astype(np.float32)
    inference_depth_image *= 0.0001 # D4054
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


