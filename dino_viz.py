import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from hand_traking import detect_hand_keypoints, sample_points_on_line
from dino_functions import Dinov2

if __name__ == "__main__":
    demo_image = cv2.imread("demo.jpeg")
    demo_image_hand = cv2.imread("demo_hand.jpeg")
    inference_image = cv2.imread("inference.jpeg")


    demo_image = cv2.resize(demo_image, (demo_image_hand.shape[1], demo_image_hand.shape[0]))
    inference_image = cv2.resize(inference_image, (demo_image_hand.shape[1], demo_image_hand.shape[0]))

    print(demo_image.shape)
    print(demo_image_hand.shape)
    print(inference_image.shape)

    # cv2.imshow("demo_image", demo_image)
    cv2.imshow("demo_image_hand", demo_image_hand)
    # cv2.imshow("inference_image", inference_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    keypoints = detect_hand_keypoints(demo_image_hand)
    print("keypoints: ", keypoints)
    

    # {0: {'hand': 'Left', 'thumb_tip': (298, 215), 'thumb_ip': (335, 197)}}
    samples_poitns = np.linspace(keypoints[0]['thumb_tip'], keypoints[0]['thumb_ip'], 10)
    # print(samples_poitns)
    sampled_points = [[int(x[0]), int(x[1])] for x in samples_poitns]
    print(sampled_points)

    demo_copy = demo_image
    for point in sampled_points:
        cv2.circle(demo_copy, tuple(point), 5, (0, 0, 255), -1)
    cv2.imshow("demo_copy", demo_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dino = Dinov2()

    demo_image_tensor, demo_image_grid = dino.prepare_image(demo_image)
    inference_image_tensor, inference_image_grid = dino.prepare_image(inference_image)

    print("inference_tensor: ", inference_image_tensor.shape)

    demo_image_features = dino.extract_features(demo_image_tensor)
    inference_image_features = dino.extract_features(inference_image_tensor)

    matched_pixels = []
    distance_total = None
    for point in sampled_points:
        idx = dino.pixel_to_idx([point[1],point[0]], demo_image_grid, dino.patch_size)
        print("idx: ", idx)
        distance = dino.compute_feature_distance(idx, demo_image_features, inference_image_features)
        # pixel with minimum distance
        distance = np.reshape(distance, (inference_image_grid[0], inference_image_grid[1]))
        distance = cv2.resize(distance, (inference_image_tensor.shape[2], inference_image_tensor.shape[1]), interpolation=cv2.INTER_CUBIC)
        
        if distance_total is None:
            distance_total = distance
        else:
            distance_total += distance
        
        print("distance: ", distance.shape)
        pixel = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
        matched_pixels.append(pixel)
        print("pixel: ", pixel)

    inference_copy = inference_image.copy()
    for point in matched_pixels:
        cv2.circle(inference_copy, tuple(point), 2, (0, 0, 255), -1)
    cv2.imshow("inference_copy", inference_copy)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    plt.imshow(inference_image)
    plt.imshow(distance_total, alpha=0.5)
    plt.colorbar()
    plt.show()


