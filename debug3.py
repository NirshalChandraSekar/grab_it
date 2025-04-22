import open3d as o3d
import cv2
import numpy as np
from hand_traking import detect_hand_keypoints, sample_points_on_line
from dino_functions import Dinov2
import torch
from segmentation import image_segmentation
import matplotlib.pyplot as plt

def create_colored_sphere(color, center, radius=0.005):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere


def show_mask(mask,ax,random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # draw a contour around the mask
    
    ax.imshow(mask_image)
def crop_mask(mask, start_y, start_x, target_height, target_width):
    """Crop the mask to match the image dimensions after cropping."""
    return mask[start_y:start_y + target_height, start_x:start_x + target_width]

def center_crop(img, target_height, target_width):
    """Crop an image equally from all sides to reach target dimensions."""
    h, w = img.shape[:2]
    start_y = (h - target_height) // 2
    start_x = (w - target_width) // 2
    return img[start_y : start_y + target_height, start_x : start_x + target_width]

def adjust_points(points, orig_height, orig_width, crop_height, crop_width):
    """Adjust 2D points to match the cropped region."""
    start_y = (orig_height - crop_height) // 2
    start_x = (orig_width - crop_width) // 2
    adjusted_points = [(x - start_x, y - start_y) for (x, y) in points]
    return adjusted_points

def create_image_with_overlay(image, mask, alpha=0.4):
    """
    Create a new image with mask overlay baked in
    
    Args:
        image: RGB image
        mask: Binary mask 
        alpha: Transparency of the overlay (0-1)
    
    Returns:
        Image with overlay
    """
    # Make sure mask is in correct format [H, W]
    if len(mask.shape) == 3:
        if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:  # If mask is [C, H, W]
            # Transpose to [H, W, C]
            mask = np.transpose(mask, (1, 2, 0))
        
        if mask.shape[2] > 1:  # If mask has multiple channels
            # Sum across channels and normalize
            mask_single = np.sum(mask, axis=2) > 0
        else:
            mask_single = mask[:, :, 0] > 0
    else:
        mask_single = mask > 0
    
    # Ensure mask is the same shape as the image
    h, w = image.shape[:2]
    if mask_single.shape[0] != h or mask_single.shape[1] != w:
        # Resize mask to match image dimensions
        mask_single = cv2.resize(mask_single.astype(np.uint8), (w, h)) > 0
    
    # Create a colored overlay
    color = np.array([255/255, 144/255, 30/255])  # RGB color
    
    # Make sure image is in the right format
    if image.dtype != np.float32:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.copy()
    
    # Blend the overlay with the image
    result = image_float.copy()
    
    for c in range(3):
        channel = result[:, :, c]
        channel[mask_single] = channel[mask_single] * (1 - alpha) + color[c] * alpha
        result[:, :, c] = channel
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)

    # Draw contours around the mask
    binary_mask = mask_single.astype(np.uint8)  # Convert mask to binary format
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), thickness=5)  # Red contours with thickness 2

    
    return result


if __name__ == "__main__":
    
    demo_image = cv2.imread('resources/tool_2/demo.jpg')
    demo_image = cv2.resize(demo_image, (demo_image.shape[1] // 5, demo_image.shape[0] // 5))
    demo_hand_image = cv2.imread('resources/tool_2/demo_hand.jpg')
    demo_hand_image = cv2.resize(demo_hand_image, (demo_hand_image.shape[1] // 5, demo_hand_image.shape[0] // 5))

    # keypoints = detect_hand_keypoints(demo_hand_image)
    # np.save('resources/tool_2/keypoints.npy', keypoints)

    keypoints = np.load('resources/tool_2/keypoints.npy', allow_pickle=True).item()
    keypoint_overlay = demo_image.copy()



    # Feature Matching using DINO
    dino = Dinov2()

    # checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    # config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # segmentor = image_segmentation(checkpoint=checkpoint, config=config, device=device)
    # mask = segmentor.segment_image(demo_image)

    # np.save('resources/tool_2/mask.npy', mask)

    mask = np.load('resources/tool_2/mask.npy', allow_pickle=True)
    mask = np.transpose(mask, (1, 2, 0))
    


    contact_point_2d = {}
    directional_point_2d = {}
    final_image = demo_image.copy()
    for key in keypoints:
        contact_point_2d[key] = [int(abs(keypoints[key]["thumb_tip"][0]+keypoints[key]["thumb_ip"][0])/2),
                              int(abs(keypoints[key]["thumb_tip"][1]+keypoints[key]["thumb_ip"][1])/2)]
        directional_point_2d[key] = keypoints[key]["thumb_tip"]
        cv2.circle(final_image, (contact_point_2d[key][0], contact_point_2d[key][1]), 5, (0, 255, 0), -1)
        cv2.circle(final_image, (directional_point_2d[key][0], directional_point_2d[key][1]), 5, (255, 0, 0), -1)
        cv2.circle(final_image, (keypoints[key]["thumb_ip"][0], keypoints[key]["thumb_ip"][1]), 5, (0, 0, 255), -1)


    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    inference_image = np.load('resources/tool_2/inference_color_image_127122270512.npy')
    cv2.imwrite('resources/tool_2/cropped_inference_image.png', inference_image)
    demo_h, demo_w = demo_image.shape[:2]
    inference_h, inference_w = inference_image.shape[:2]

    


    # feature extraction
    sampled_points_2d = {}
    for key in keypoints:
        sampled_points_2d[key] = sample_points_on_line(
            keypoints[key]["thumb_ip"], keypoints[key]["thumb_tip"], mask, n=20
        )
        # for point in sampled_points_2d[key]:
        #     color = (0, 255, 255)
        #     cv2.circle(final_image, (point[0], point[1]), 5, color, -1)

    # cv2.circle(final_image, (contact_point_2d[0][0], contact_point_2d[key][1]), 5, (0, 255, 0), -1)
    # cv2.circle(final_image, (directional_point_2d[0][0], directional_point_2d[key][1]), 5, (0, 0, 255), -1)
    # cv2.circle(final_image, (keypoints[key]["thumb_ip"][0], keypoints[key]["thumb_ip"][1]), 5, (255, 0, 0), -1)

    # cv2.imshow("Final Image", final_image)
    # plt.imshow(final_image)
    # mask = np.transpose(mask, (2, 0, 1))
    # show_mask(mask, plt.gca(), random_color=False)
    # # add circles im plae of contact point, directional point and thumb ip in this matplotlib image
    # for key in sampled_points_2d:
    #     for point in sampled_points_2d[key]:
    #         color = 'yellow'
    #         plt.scatter(point[0], point[1], c=color, s=10)

    # plt.scatter(contact_point_2d[0][0], contact_point_2d[0][1], c='green', s=10, label='Contact Point')
    # plt.scatter(directional_point_2d[0][0], directional_point_2d[0][1], c='blue', s=10, label='Directional Point')
    # plt.scatter(keypoints[0]["thumb_ip"][0], keypoints[0]["thumb_ip"][1], c='red', s=10, label='Thumb IP')
    
    # plt.show()

    final_image = create_image_with_overlay(final_image, mask, alpha=0.4)
    cv2.circle(final_image, (contact_point_2d[0][0], contact_point_2d[0][1]), 5, (0, 255, 0), -1)
    cv2.circle(final_image, (directional_point_2d[0][0], directional_point_2d[0][1]), 5, (0, 0, 255), -1)
    # cv2.imshow("Final Image", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    dino = Dinov2()
    camera_serials = [130322273305, 127122270512, 126122270722]
    
    demo_image_tensor, demo_image_grid = dino.prepare_image(demo_image)
    demo_image_features = dino.extract_features(demo_image_tensor)
    inference_image_tensor, inference_image_grid = dino.prepare_image(inference_image)
    inference_image_features = dino.extract_features(inference_image_tensor)

    demo_indices = [dino.pixel_to_idx([point[1], point[0]], demo_image_grid, dino.patch_size) for point in sampled_points_2d[0]]
    demo_contact_point_index = dino.pixel_to_idx([contact_point_2d[0][1], contact_point_2d[0][0]], demo_image_grid, dino.patch_size)
    demo_directional_point_index = dino.pixel_to_idx([directional_point_2d[0][1], directional_point_2d[0][0]], demo_image_grid, dino.patch_size)

    corresponding_points = {0: []}
    for point_index in demo_indices:
        distance = dino.compute_feature_distance(point_index, demo_image_features, inference_image_features)
        distance = np.reshape(distance, (inference_image_grid[0], inference_image_grid[1]))
        distance = cv2.resize(distance, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        threshold = np.percentile(distance, 0.005)
        distance_mask = (distance < threshold).astype(np.uint8)

        mean_point = np.mean(np.argwhere(distance_mask > 0), axis=0)
        corresponding_points[0].append(np.array([int(mean_point[0]), int(mean_point[1])]))

    corresponding_contact_point = dino.compute_feature_distance(demo_contact_point_index, demo_image_features, inference_image_features)
    corresponding_contact_point = np.reshape(corresponding_contact_point, (inference_image_grid[0], inference_image_grid[1]))
    corresponding_contact_point = cv2.resize(corresponding_contact_point, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(corresponding_contact_point, 0.005)
    corresponding_contact_point_mask = (corresponding_contact_point < threshold).astype(np.uint8)
    mean_contact_point = np.mean(np.argwhere(corresponding_contact_point_mask > 0), axis=0)
    corresponding_contact_point = np.array([int(mean_contact_point[0]), int(mean_contact_point[1])])

    corresponding_directional_point = dino.compute_feature_distance(demo_directional_point_index, demo_image_features, inference_image_features)
    corresponding_directional_point = np.reshape(corresponding_directional_point, (inference_image_grid[0], inference_image_grid[1]))
    corresponding_directional_point = cv2.resize(corresponding_directional_point, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(corresponding_directional_point, 0.005)
    corresponding_directional_point_mask = (corresponding_directional_point < threshold).astype(np.uint8)
    mean_directional_point = np.mean(np.argwhere(corresponding_directional_point_mask > 0), axis=0)
    corresponding_directional_point = np.array([int(mean_directional_point[0]), int(mean_directional_point[1])])


    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)
    inference_depth = np.load('resources/tool_2/inference_depth_image_127122270512.npy').astype(np.float32)*0.00025
    cx, cy, fx, fy = np.load('resources/tool_2/camera_intrinsic_127122270512.npy')

    



    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        inference_image.shape[1],  # Image width
        inference_image.shape[0],  # Image height
        fx, fy, cx, cy
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(inference_image),
        o3d.geometry.Image(inference_depth),
        depth_scale=1.0,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics,
    )
    pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    
    geometries = [pcd]

    for point in corresponding_points[0]:
        y,x = point
        x_3d = (x - cx) * inference_depth[y, x] / fx
        y_3d = (y - cy) * inference_depth[y, x] / fy
        z_3d = inference_depth[y, x]
        geometries.append(create_colored_sphere([1, 1, 0], [x_3d, -y_3d, -z_3d], radius=0.009))


    if inference_depth[corresponding_contact_point[1], corresponding_contact_point[0]] != 0:
        y,x = corresponding_contact_point
        x_3d = (x - cx) * inference_depth[y, x] / fx
        y_3d = (y - cy) * inference_depth[y, x] / fy
        z_3d = inference_depth[y, x]
        geometries.append(create_colored_sphere([0, 1, 0], [x_3d, -y_3d, -z_3d], radius = 0.01))
        print('Contact Point:', [x_3d, -y_3d, -z_3d])   

    if inference_depth[corresponding_directional_point[1], corresponding_directional_point[0]] != 0:
        y,x = corresponding_directional_point
        x_3d = (x - cx) * inference_depth[y, x] / fx
        y_3d = (y - cy) * inference_depth[y, x] / fy    
        z_3d = inference_depth[y, x]
        geometries.append(create_colored_sphere([0, 0, 1], [x_3d, -y_3d, -z_3d], radius=0.01))
        print('Directional Point:', [x_3d, -y_3d, -z_3d])


    o3d.visualization.draw_geometries(geometries)

     
























    # Get minimum dimensions
    # min_height = min(final_image.shape[0], inference_image.shape[0])
    # min_width = min(final_image.shape[1], inference_image.shape[1])

    min_height = int(inference_image.shape[0]//2)
    min_width = int(inference_image.shape[1]//2.8)


    # Crop both images (centered)
    final_cropped = center_crop(final_image, min_height, min_width)
    inference_cropped = center_crop(inference_image, min_height, min_width)

    # Calculate crop offsets
    demo_start_y = (final_image.shape[0] - min_height) // 2
    demo_start_x = (final_image.shape[1] - min_width) // 2
    inf_start_y = (inference_image.shape[0] - min_height) // 2
    inf_start_x = (inference_image.shape[1] - min_width) // 2

    # Adjust points to the cropped region
    adjusted_demo_points = []
    for point in sampled_points_2d[0]:
        x, y = point
        adjusted_x = x - demo_start_x
        adjusted_y = y - demo_start_y
        if 0 <= adjusted_x < min_width and 0 <= adjusted_y < min_height:
            adjusted_demo_points.append((adjusted_x, adjusted_y))

    adjusted_inf_points = []
    for point in corresponding_points[0]:
        y, x = point  # Note: Your points seem to be in (y, x) format
        adjusted_x = x - inf_start_x
        adjusted_y = y - inf_start_y
        if 0 <= adjusted_x < min_width and 0 <= adjusted_y < min_height:
            adjusted_inf_points.append((adjusted_x, adjusted_y))

    # Also adjust contact and directional points
    adjusted_contact_point = (
        corresponding_contact_point[1] - inf_start_x,
        corresponding_contact_point[0] - inf_start_y
    )
    adjusted_directional_point = (
        corresponding_directional_point[1] - inf_start_x,
        corresponding_directional_point[0] - inf_start_y
    )

    adjusted_demo_contact_point = (
        contact_point_2d[0][0] - demo_start_x,
        contact_point_2d[0][1] - demo_start_y
    )
    adjusted_demo_directional_point = (
        directional_point_2d[0][0] - demo_start_x,
        directional_point_2d[0][1] - demo_start_y
    )


    # Stack images vertically
    combined_image = np.vstack((final_cropped, inference_cropped))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(combined_image)

    # Offset for inference image points vertically
    offset_y = final_cropped.shape[0]

    # Draw correspondences - make sure we have the same number of points
    min_points = min(len(adjusted_demo_points), len(adjusted_inf_points))
    for i in range(min_points):
        demo_x, demo_y = adjusted_demo_points[i]
        inf_x, inf_y = adjusted_inf_points[i]
        plt.plot([demo_x, inf_x], [demo_y, inf_y + offset_y], 'y-', lw=0.5, alpha=0.3)

    # Mark points (yellow: demo, inference)
    demo_x_points = [x for x, y in adjusted_demo_points]
    demo_y_points = [y for x, y in adjusted_demo_points]
    plt.scatter(demo_x_points, demo_y_points, c='yellow', s=5, label='Demo points')

    inf_x_points = [x for x, y in adjusted_inf_points]
    inf_y_points = [y + offset_y for x, y in adjusted_inf_points]
    plt.scatter(inf_x_points, inf_y_points, c='yellow', s=5, label='Inference points')

    # Also mark the contact and directional points
    if 0 <= adjusted_contact_point[0] < min_width and 0 <= adjusted_contact_point[1] < min_height:
        plt.scatter(adjusted_contact_point[0], adjusted_contact_point[1] + offset_y, 
                    c='green', s=70, marker='*', label='Contact point')

    if 0 <= adjusted_directional_point[0] < min_width and 0 <= adjusted_directional_point[1] < min_height:
        plt.scatter(adjusted_directional_point[0], adjusted_directional_point[1] + offset_y, 
                    c='blue', s=70, marker='*', label='Directional point')

    if 0 <= adjusted_demo_contact_point[0] < min_width and 0 <= adjusted_demo_contact_point[1] < min_height:
        plt.scatter(adjusted_demo_contact_point[0], adjusted_demo_contact_point[1], 
                    c='green', s=70, marker='*', label='Demo Contact point')

    if 0 <= adjusted_demo_directional_point[0] < min_width and 0 <= adjusted_demo_directional_point[1] < min_height:
        plt.scatter(adjusted_demo_directional_point[0], adjusted_demo_directional_point[1], 
                    c='blue', s=70, marker='*', label='Demo Directional point')

    ax.axis('off')
    plt.title("Point Correspondences between Demo (top) and Inference (bottom) Images", pad=20)
    plt.tight_layout()
    plt.show()



    # # Stack images horizontally
    # combined_image = np.hstack((final_cropped, inference_cropped))

    # # Plotting
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.imshow(combined_image)

    # # Offset for inference image points
    # offset_x = final_cropped.shape[1]

    # # Draw correspondences - make sure we have the same number of points
    # min_points = min(len(adjusted_demo_points), len(adjusted_inf_points))
    # for i in range(min_points):
    #     demo_x, demo_y = adjusted_demo_points[i]
    #     inf_x, inf_y = adjusted_inf_points[i]
    #     plt.plot([demo_x, inf_x + offset_x], [demo_y, inf_y], 'y-', lw=0.5, alpha=0.3)

    # # Mark points (red: demo, blue: inference)
    # demo_x_points = [x for x, y in adjusted_demo_points]
    # demo_y_points = [y for x, y in adjusted_demo_points]
    # plt.scatter(demo_x_points, demo_y_points, c='yellow', s=5, label='Demo points')

    # inf_x_points = [x + offset_x for x, y in adjusted_inf_points]
    # inf_y_points = [y for x, y in adjusted_inf_points]
    # plt.scatter(inf_x_points, inf_y_points, c='yellow', s=5, label='Inference points')

    # # Also mark the contact and directional points
    # if 0 <= adjusted_contact_point[0] < min_width and 0 <= adjusted_contact_point[1] < min_height:
    #     plt.scatter(adjusted_contact_point[0] + offset_x, adjusted_contact_point[1], 
    #                 c='green', s=70, marker='*', label='Contact point')

    # if 0 <= adjusted_directional_point[0] < min_width and 0 <= adjusted_directional_point[1] < min_height:
    #     plt.scatter(adjusted_directional_point[0] + offset_x, adjusted_directional_point[1], 
    #                 c='blue', s=70, marker='*', label='Directional point')

    # if 0 <= adjusted_demo_contact_point[0] < min_width and 0 <= adjusted_demo_contact_point[1] < min_height:
    #     plt.scatter(adjusted_demo_contact_point[0], adjusted_demo_contact_point[1], 
    #                 c='green', s=70, marker='*', label='Demo Contact point')
    # if 0 <= adjusted_demo_directional_point[0] < min_width and 0 <= adjusted_demo_directional_point[1] < min_height:
    #     plt.scatter(adjusted_demo_directional_point[0], adjusted_demo_directional_point[1], 
    #                 c='blue', s=70, marker='*', label='Demo Directional point')

    # # show_mask(cropped_mask, ax, random_color=False)

    # # plt.legend(loc='upper right')
    # ax.axis('off')
    # plt.title("Point Correspondences between Demo and Inference Images", pad=20)
    # plt.tight_layout()
    # plt.show()


