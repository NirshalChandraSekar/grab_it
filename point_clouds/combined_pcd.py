import open3d as o3d
import numpy as np
import cv2

"""
Transformation Matrix from Camera 126122270722 to Camera 130322273305:
[[-0.6702651  -0.40290053  0.62323018 -0.14233934]
 [ 0.33793223  0.58197548  0.73966637 -0.21903271]
 [-0.66071666  0.70638212 -0.25392478  0.46625483]
 [ 0.          0.          0.          1.        ]]
 """
"""
Transformation Matrix from Camera 127122270512 to Camera 130322273305:
[[-0.72935441  0.305626   -0.61207425  0.1601601 ]
 [-0.426135    0.49696257  0.75593464 -0.25791868]
 [ 0.53521127  0.81217052 -0.2322239   0.45498251]
 [ 0.          0.          0.          1.        ]]
 """
class GeneratePCD:
    def __init__(self):
        self.camera_serials = ['130322273305', '126122270722', '127122270512']
        self.idx_to_serial = {0: '130322273305', 1: '126122270722', 2: '127122270512'}
        self.serial_to_idx = {"130322273305": 0, "126122270722": 1, "127122270512": 2}
        
        self.transformation_1to0 = np.array([
            [-0.6702651,  -0.40290053,  0.62323018, -0.14233934],
            [ 0.33793223,  0.58197548,  0.73966637, -0.21903271],
            [-0.66071666,  0.70638212, -0.25392478,  0.46625483],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])

        self.transformation_2to0 = np.array([
            [-0.72935441,  0.305626  , -0.61207425,  0.1601601 ],
            [-0.426135  ,  0.49696257,  0.75593464, -0.25791868],
            [ 0.53521127,  0.81217052, -0.2322239 ,  0.45498251],
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
                fx=intrinsic_dict[key][2],
                fy=intrinsic_dict[key][3],
                cx=intrinsic_dict[key][0],
                cy=intrinsic_dict[key][1],
            )
            transformation = self.transformation_dict[key]

            depth_trunc_ = None
            if self.serial_to_idx[key] == 1 or self.serial_to_idx[key] == 2:
                depth_trunc_ = 0.8
            else:
                depth_trunc_ = 0.8
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                depth_scale=1,#/0.0001,#1/0.0001,
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
        
        o3d.io.write_point_cloud("point_clouds/combined_point_cloud_test.ply", self.combined_pcd)



if __name__ == "__main__":
    color_image_dict = {
        '130322273305': np.load('resources/green_pouch_0/inference_color_image_130322273305.npy'),
        '126122270722': np.load('resources/green_pouch_0/inference_color_image_126122270722.npy'),
        '127122270512': np.load('resources/green_pouch_0/inference_color_image_127122270512.npy'),
    }
    depth_image_dict = {
        "130322273305": np.load("resources/green_pouch_0/inference_depth_image_130322273305.npy").astype(np.float32)*0.0001,
        "126122270722": np.load("resources/green_pouch_0/inference_depth_image_126122270722.npy").astype(np.float32)*0.0001,
        "127122270512": np.load("resources/green_pouch_0/inference_depth_image_127122270512.npy").astype(np.float32)*0.0001,
    }
    intrinsic_dict = {
        "130322273305": np.load("resources/green_pouch_0/camera_intrinsic_130322273305.npy"),
        "126122270722": np.load("resources/green_pouch_0/camera_intrinsic_126122270722.npy"),
        "127122270512": np.load("resources/green_pouch_0/camera_intrinsic_127122270512.npy"),
    }

    pcd_generator = GeneratePCD()
    pcd_generator.generate_pcd(color_image_dict, depth_image_dict, intrinsic_dict)
    pcd_generator.visualize_pcd()

    



