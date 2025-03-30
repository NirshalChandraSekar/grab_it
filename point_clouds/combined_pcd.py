import open3d as o3d
import numpy as np
import cv2

"""
Transformation Matrix from Camera 126122270722 to Camera 130322273305:
[[-0.66429853 -0.40768787  0.62649666 -0.32571419]
 [ 0.33387348  0.58805779  0.73669298 -0.49335773]
 [-0.66875703  0.69855468 -0.25452975  1.03414807]
 [ 0.          0.          0.          1.        ]]
 """
"""
Transformation Matrix from Camera 127122270512 to Camera 130322273305:
[[-0.72879876  0.30043344 -0.6152984   0.36475736]
 [-0.42875943  0.50039296  0.75217833 -0.57977843]
 [ 0.53387051  0.81200162 -0.2358721   1.01002089]
 [ 0.          0.          0.          1.        ]]
 """
class GeneratePCD:
    def __init__(self):
        self.camera_serials = ['130322273305', '126122270722', '127122270512']
        self.idx_to_serial = {0: '130322273305', 1: '126122270722', 2: '127122270512'}
        self.serial_to_idx = {"130322273305": 0, "126122270722": 1, "127122270512": 2}
        
        self.transformation_1to0 = np.array([
            [-0.66429853, -0.40768787,  0.62649666, -0.32571419],
            [ 0.33387348,  0.58805779,  0.73669298, -0.49335773],
            [-0.66875703,  0.69855468, -0.25452975,  1.03414807],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        self.transformation_2to0 = np.array([
            [-0.72879876,  0.30043344, -0.6152984 ,  0.36475736],
            [-0.42875943,  0.50039296,  0.75217833, -0.57977843],
            [ 0.53387051,  0.81200162, -0.2358721 ,  1.01002089],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        

        self.transformation_dict = {
            '130322273305': np.eye(4),
            '126122270722': self.transformation_1to0,
            '127122270512': self.transformation_2to0,
        }
        
       
        self.combined_pcd = o3d.geometry.PointCloud()
        self.camera_frame = {}

    def generate_pcd(self, color_image_dict, depth_image_dict, intrinsic_dict):
        keys = list(color_image_dict.keys())
        for key in keys:
            color_image = color_image_dict[key]
            depth_image = depth_image_dict[key]
            # intrinsic = intrinsic_dict[key]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640,
                height=480,
                fx=intrinsic_dict[key][0][0],
                fy=intrinsic_dict[key][1][1],
                cx=intrinsic_dict[key][0][2],
                cy=intrinsic_dict[key][1][2],
            )
            transformation = self.transformation_dict[key]

            depth_trunc_ = None
            if self.serial_to_idx[key] == 1 or self.serial_to_idx[key] == 2:
                depth_trunc_ = 1
            else:
                depth_trunc_ = 2
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                depth_scale=1,
                depth_trunc=depth_trunc_,
                convert_rgb_to_intensity=False,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic, transformation
            )

            self.combined_pcd += pcd

    def transform_pcd(self, transformation_dict):
        for key in self.pcd.keys():
            pcd = self.pcd[key]
            transformation = transformation_dict[key]
            pcd.transform(transformation)



    def visualize_pcd(self):
        
        o3d.io.write_point_cloud("point_clouds/combined_point_cloud.ply", self.combined_pcd)



if __name__ == "__main__":
    color_image_dict = {
        '130322273305': cv2.imread("cameras/test_images/0/130322273305_color.png"),
        '126122270722': cv2.imread("cameras/test_images/0/126122270722_color.png"),
        '127122270512': cv2.imread("cameras/test_images/0/127122270512_color.png"),
    }
    depth_image_dict = {
        "130322273305": np.load("cameras/test_images/0/130322273305_depth.npy").astype(np.float32)*0.00025,
        "126122270722": np.load("cameras/test_images/0/126122270722_depth.npy").astype(np.float32)*0.00025,
        "127122270512": np.load("cameras/test_images/0/127122270512_depth.npy").astype(np.float32)*0.00025,
    }
    intrinsic_dict = {
        "130322273305": np.load("cameras/test_images/0/130322273305_intrinsic_matrix.npy"),
        "126122270722": np.load("cameras/test_images/0/126122270722_intrinsic_matrix.npy"),
        "127122270512": np.load("cameras/test_images/0/127122270512_intrinsic_matrix.npy"),
    }

    pcd_generator = GeneratePCD()
    pcd_generator.generate_pcd(color_image_dict, depth_image_dict, intrinsic_dict)
    pcd_generator.visualize_pcd()

    



