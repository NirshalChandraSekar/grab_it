from hand_traking import detect_hand_keypoints, sample_points_on_line
import cv2
import numpy as np
import torch
from inference_stream import InferenceStream, InferenceMultiCamera
import os

if __name__ == "__main__":

    test_number = "1"
    object_name = "camera_calibration"
    if not os.path.exists("resources/" + object_name +  "/" + test_number):
        os.makedirs("resources/" + object_name +  "/" + test_number)
    stream = InferenceMultiCamera()
    frames_dict = stream.get_frames()
    for key in frames_dict:
        color_image = frames_dict[key]["color"]
        depth_image = frames_dict[key]["depth"]
        intrinsic = [frames_dict[key]["intrinsics"]['ppx'],
                        frames_dict[key]["intrinsics"]['ppy'],
                        frames_dict[key]["intrinsics"]['fx'],
                        frames_dict[key]["intrinsics"]['fy']]
        distortion = [frames_dict[key]["intrinsics"]['coeffs']]
        np.save("resources/" + object_name +  "/" + test_number + "/inference_color_image_" + key + ".npy", color_image)
        np.save("resources/" + object_name +  "/" + test_number + "/inference_depth_image_" + key + ".npy", depth_image)
        np.save("resources/" + object_name +  "/" + test_number + "/camera_intrinsic_" + key + ".npy", intrinsic)
        np.save("resources/" + object_name +  "/" + test_number + "/camera_distortion_" + key + ".npy", distortion)
        cv2.imshow("color", color_image)
        cv2.imshow("depth", depth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    


    # image = cv2.imread("/home/nirshal/Downloads/IMG_1496.jpg")
    # image = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # keypoints = detect_hand_keypoints(image)
    # print(keypoints)
    # number = 3
    # color_image, depth_image, intrinsic = InferenceStream().get_frame()
    # cv2.imshow("color", color_image)
    # cv2.imshow("depth", depth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # np.save("mohit/"+str(number)+"/color_image.npy", color_image)
    # np.save("mohit/"+str(number)+"/depth_image.npy", depth_image)
    # np.save("mohit/"+str(number)+"/intrinsic.npy", intrinsic)

    # multi_camera = InferenceMultiCamera()
    # frames = multi_camera.get_frames()

    # for serial, data in frames.items():
    #     color_image = data['color']
    #     depth_image = data['depth']
    #     intrinsic = data['intrinsics']

    #     cv2.imshow(str(serial) + " color", color_image)
    #     cv2.imshow(str(serial) + " depth", depth_image)
    #     cv2.waitKey(0)
    #     # print("intrinsic type", type(intrinsic['ppx']))
    #     # print('type of model', type(intrinsic['model']))
    #     # print('type of coeffs', type(intrinsic['coeffs']))

    # cv2.destroyAllWindows()

    # np.save("mohit/multi_camera_aruco_board.npy", frames)
