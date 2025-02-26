# from hand_traking import detect_hand_keypoints, sample_points_on_line
import cv2
import numpy as np
# import torch
# from inference_stream import InferenceStream, InferenceMultiCamera


if __name__ == "__main__":
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
    # frames_dict = multi_camera.get_frames()

    # for key in frames_dict:
    #     color_image = frames_dict[key]["color"]
    #     depth_image = frames_dict[key]["depth"]
    #     intrinsic = [frames_dict[key]["intrinsics"]['ppx'],
    #                     frames_dict[key]["intrinsics"]['ppy'],
    #                     frames_dict[key]["intrinsics"]['fx'],
    #                     frames_dict[key]["intrinsics"]['fy']]
        
    #     np.save("mohit/robot_camera_calibration/color_image_"+str(key)+".npy", color_image) 
    #     np.save("mohit/robot_camera_calibration/depth_image_"+str(key)+".npy", depth_image)
    #     cv2.imshow("color", color_image)
    #     cv2.imshow("depth", depth_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     np.save("mohit/robot_camera_calibration/intrinsic_"+str(key)+".npy", intrinsic)
    #     print("Saved Camera "+str(key))
        
    ## Load intinsics
    intrin = np.load('resources/camera_calibration/camera_distortion_130322273305.npy'    )
    print("Intrinsics: ", intrin)

    # cv2.destroyAllWindows()

    # np.save("mohit/multi_camera_pouch.npy", frames)
