from hand_traking import detect_hand_keypoints, sample_points_on_line
from inference_stream import InferenceStream, InferenceMultiCamera
from viz import pca_2d, pca_3d, get_gt, visualize_gripper
from segmentation import image_segmentation
from features import find_best_camera
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
    object_name = "green_pouch_0"
    # stream = InferenceMultiCamera()


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
    print("Keypoints: ", keypoints)
    keypoint_overlay = demo_image.copy()
    contact_point_2d = {}
    directional_point_2d = {}

    for key in keypoints:
        color = (0, 0, 255) if key == 0 else (255, 0, 0)
        contact_point_2d[key] = [int(abs(keypoints[key]["thumb_tip"][0]+keypoints[key]["thumb_ip"][0])/2),
                              int(abs(keypoints[key]["thumb_tip"][1]+keypoints[key]["thumb_ip"][1])/2)]
        directional_point_2d[key] = keypoints[key]["thumb_tip"]
        cv2.circle(keypoint_overlay, keypoints[key]["thumb_tip"], 3, color, -1)
        cv2.circle(keypoint_overlay, keypoints[key]["thumb_ip"], 3, color,  -1)
        cv2.circle(keypoint_overlay, contact_point_2d[key], 3, (0, 255, 0), -1)

    

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
        # for point in sampled_points_2d[key]:
        #     cv2.circle(keypoint_overlay, point, 3, (0, 255, 0), -1)

    # display_image(keypoint_overlay, "Sampled Points Overlay")



    '''
    Record Inference using the realsense camera
    '''
    # check = input("Do you want to record the inference? (1 for yes, 0 for no): ")
    check = "0"
    if check == "1":
        _ = input("Verify if camera is connected and press enter to continue")
        # stream = InferenceStream()
        # color_image, depth_image, intrinsic = stream.get_frame()
        # display_image(color_image, "Inference Image")
        # display_image(depth_image, "Inference Depth Image")
        # np.save("resources/" + object_name + "/inference_color_image.npy", color_image)
        # np.save("resources/" + object_name + "/inference_depth_image.npy", depth_image)
        # np.save("resources/" + object_name + "/camera_intrinsic.npy", intrinsic)
        
        frames_dict = stream.get_frames()
        for key in frames_dict:
            color_image = frames_dict[key]["color"]
            depth_image = frames_dict[key]["depth"]
            intrinsic = [frames_dict[key]["intrinsics"]['ppx'],
                         frames_dict[key]["intrinsics"]['ppy'],
                         frames_dict[key]["intrinsics"]['fx'],
                         frames_dict[key]["intrinsics"]['fy']]
            distortion = [frames_dict[key]["intrinsics"]['coeffs']]
            np.save("resources/" + object_name + "/inference_color_image_" + key + ".npy", color_image)
            np.save("resources/" + object_name + "/inference_depth_image_" + key + ".npy", depth_image)
            np.save("resources/" + object_name + "/camera_intrinsic_" + key + ".npy", intrinsic)
            np.save("resources/" + object_name + "/camera_distortion_" + key + ".npy", distortion)
    



    '''
    Run the dino functions for feature extraction
    '''

    dino = Dinov2()
    camera_serials = [130322273305, 127122270512, 126122270722]

    best_serial, distance_map, inference_contact_point, inference_directional_point = find_best_camera(dino, demo_image, sampled_points_2d, contact_point_2d, directional_point_2d, object_name, camera_serials)

    print("Best Serial: ", best_serial)
    # inference_color_image = np.load("resources/" + object_name + "/inference_color_image_" + str(best_serial) + ".npy")
    # inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)

    inference_color_images = {}

    for key in best_serial:
        serial = best_serial[key]
        inference_color_images[key] = np.load("resources/" + object_name + "/inference_color_image_" + str(serial) + ".npy")
        inference_color_images[key] = cv2.cvtColor(inference_color_images[key], cv2.COLOR_BGR2RGB)




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

    print("Color of key 0 is red and key 1 is blue")

    display_image(keypoint_overlay, "Hand Keypoints Overlay")


    # intrinsics = np.load("resources/" + object_name + "/camera_intrinsic_" + str(best_serial) + ".npy")
    # inference_depth_image = np.load("resources/" + object_name + "/inference_depth_image_" + str(best_serial) + ".npy")
    # inference_color_image = np.load("resources/" + object_name + "/inference_color_image_" + str(best_serial) + ".npy")
    # inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)
    # inference_depth_image = inference_depth_image.astype(np.float32)
    # inference_depth_image *= 0.0001 # D4054
    # # inference_depth_image *= 0.00025

    # gt_grasp_axes = {}
    # for key in important_pixels:
    #     gt_grasp_axes[key] = get_gt(inference_color_image, inference_depth_image, intrinsics)
    # print("GT Grasp Axes: ", gt_grasp_axes)

    gt_grasp_axes = {}
    inference_color_images = {}
    inference_depth_images = {}
    intrinsics = {}

    for key in best_serial:
        serial = best_serial[key]

        # Load intrinsic parameters, depth image, and color image for each hand
        intrinsics[key] = np.load(f"resources/{object_name}/camera_intrinsic_{serial}.npy")
        inference_depth_images[key] = np.load(f"resources/{object_name}/inference_depth_image_{serial}.npy").astype(np.float32)
        inference_color_images[key] = np.load(f"resources/{object_name}/inference_color_image_{serial}.npy")

        # Convert images appropriately
        inference_color_images[key] = cv2.cvtColor(inference_color_images[key], cv2.COLOR_BGR2RGB)
        inference_depth_images[key] *= 0.0001  # D4054
        # inference_depth_images[key] *= 0.00025  # Uncomment if needed

        # Get GT grasp axes for each hand
        gt_grasp_axes[key] = get_gt(inference_color_images[key], inference_depth_images[key], intrinsics[key])

    print("GT Grasp Axes: ", gt_grasp_axes)

    # np.save(f"resources/{object_name}/gt_grasp_axes_{object_name}.npy", gt_grasp_axes)


    

    '''
    Compute 3D PCA
    '''
    
    
    # grasp_axes = pca_3d(important_pixels, 
    #        intrinsics, 
    #        inference_depth_image, 
    #        inference_color_image,
    #        inference_contact_point,
    #        inference_directional_point)
    
    # print("Grasp Axes: ", grasp_axes)
    # np.save("resources/"+object_name+"/grasp_pose_"+object_name+".npy", grasp_axes)

    grasp_axes = {}
    for key in best_serial:
        serial = best_serial[key]

        # Compute PCA-based grasp axes for each hand separately
        grasp_axes[key] = pca_3d(
            important_pixels[key], 
            intrinsics[key], 
            inference_depth_images[key], 
            inference_color_images[key], 
            inference_contact_point[key], 
            inference_directional_point[key]
        )

        # print(f"Grasp Axes for Hand {key}: ", grasp_axes[key])

    # Save grasp axes separately for each hand
    print("Grasp Axes: ", grasp_axes)
    # np.save(f"resources/{object_name}/grasp_pose_{object_name}.npy", grasp_axes)

        
    
    

    '''
    Computer Errors
    '''

    # Error Between the Center Points
    center_error = {}
    for key in gt_grasp_axes:
        center_gt = gt_grasp_axes[key][0:3,3]
        center_pred = grasp_axes[key][0:3,3]
        center_error[key] = np.linalg.norm(center_gt - center_pred)

    # Error Between the x axis
    approach_axis_error = {}
    for key in gt_grasp_axes:
        gt_x = gt_grasp_axes[key][0:3,0]
        pred_x = grasp_axes[key][0:3,0]
        # approach_axis_error[key] = np.arccos(np.dot(gt_x, pred_x)/(np.linalg.norm(gt_x)*np.linalg.norm(pred_x)))
        approach_axis_error[key] = np.arccos(np.dot(gt_x, pred_x))
        # convert to degrees
        approach_axis_error[key] = np.degrees(approach_axis_error[key])

    print("Center Error: ", center_error)
    print("Approach Axis Error: ", approach_axis_error)


    '''
    Visualize the gripper
    '''

    # visualize_gripper(inference_color_image, inference_depth_image, intrinsics, grasp_axes)
    

