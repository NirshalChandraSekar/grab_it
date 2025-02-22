from hand_traking import detect_hand_keypoints, sample_points_on_line
import cv2
import numpy as np
import torch

if __name__ == "__main__":
    image = cv2.imread("/home/nirshal/Downloads/IMG_1496.jpg")
    image = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    keypoints = detect_hand_keypoints(image)
    print(keypoints)