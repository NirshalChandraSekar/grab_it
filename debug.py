from hand_traking import detect_hand_keypoints, sample_points_on_line
import cv2
import numpy as np
import torch
from inference_stream import InferenceStream
if __name__ == "__main__":
    # image = cv2.imread("/home/nirshal/Downloads/IMG_1496.jpg")
    # image = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # keypoints = detect_hand_keypoints(image)
    # print(keypoints)
    number = 3
    color_image, depth_image, intrinsic = InferenceStream().get_frame()
    cv2.imshow("color", color_image)
    cv2.imshow("depth", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    np.save("mohit/"+str(number)+"/color_image.npy", color_image)
    np.save("mohit/"+str(number)+"/depth_image.npy", depth_image)
    np.save("mohit/"+str(number)+"/intrinsic.npy", intrinsic)

