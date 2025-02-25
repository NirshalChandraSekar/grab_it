import numpy as np
import cv2



def find_best_camera(dino, demo_image, sampled_points_2d, contact_point_2d, directional_point_2d, object_name, camera_serials):
    demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
    
    demo_image_tensor, demo_image_grid = dino.prepare_image(demo_image)
    demo_image_features = dino.extract_features(demo_image_tensor)

    indices_sampled_points_2d = {}
    demo_contact_point_index = {}
    demo_directional_point_index = {}
    
    for key in sampled_points_2d:
        indices_sampled_points_2d[key] = [dino.pixel_to_idx([point[1], point[0]], demo_image_grid, dino.patch_size) for point in sampled_points_2d[key]]
        demo_contact_point_index[key] = dino.pixel_to_idx([contact_point_2d[key][1], contact_point_2d[key][0]], demo_image_grid, dino.patch_size)
        demo_directional_point_index[key] = dino.pixel_to_idx([directional_point_2d[key][1], directional_point_2d[key][0]], demo_image_grid, dino.patch_size)

    best_serial = None
    min_distance = float('inf')
    best_inference_contact_point = None
    best_inference_directional_point = None
    best_distance_map = None

    for serial in camera_serials:
        inference_color_image = np.load(f"resources/{object_name}/inference_color_image_{serial}.npy")
        inference_color_image = cv2.cvtColor(inference_color_image, cv2.COLOR_BGR2RGB)

        inference_image_tensor, inference_image_grid = dino.prepare_image(inference_color_image)
        inference_image_features = dino.extract_features(inference_image_tensor)

        total_distance = 0
        dist_map = {}
        contact_pt = {}
        directional_pt = {}

        for key in indices_sampled_points_2d:
            dist_map[key] = None
            for point_index in indices_sampled_points_2d[key]:
                distance = dino.compute_feature_distance(point_index, demo_image_features, inference_image_features)
                distance = np.reshape(distance, (inference_image_grid[0], inference_image_grid[1]))
                distance = cv2.resize(distance, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
                threshold = np.percentile(distance, 0.005)
                distance_mask = (distance < threshold).astype(np.uint8)
                distance *= distance_mask

                dist_map[key] = distance if dist_map[key] is None else np.logical_or(dist_map[key], distance > 0)
                total_distance += np.sum(distance)

            for point_type, point_index in zip(['contact', 'directional'], [demo_contact_point_index[key], demo_directional_point_index[key]]):
                point_distance = dino.compute_feature_distance(point_index, demo_image_features, inference_image_features)
                point_distance = np.reshape(point_distance, (inference_image_grid[0], inference_image_grid[1]))
                point_distance = cv2.resize(point_distance, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
                threshold = np.percentile(point_distance, 0.005)
                point_distance_mask = (point_distance < threshold).astype(np.uint8)
                point_distance *= point_distance_mask
                
                if point_type == 'contact':
                    contact_pt[key] = np.mean(np.argwhere(point_distance_mask > 0), axis=0)
                else:
                    directional_pt[key] = np.mean(np.argwhere(point_distance_mask > 0), axis=0)
                
                total_distance += np.sum(point_distance)

        if total_distance < min_distance:
            min_distance = total_distance
            best_serial = serial
            best_distance_map = dist_map
            best_inference_contact_point = contact_pt
            best_inference_directional_point = directional_pt

    print("Best Serial:", best_serial)
    return best_serial, best_distance_map, best_inference_contact_point, best_inference_directional_point
