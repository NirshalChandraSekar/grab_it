import pyrealsense2 as rs
import numpy as np
import cv2


class InferenceStream:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)  # Align depth to color frame

    def get_frame(self):
        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)  # Align frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        intrinsic = [color_frame.profile.as_video_stream_profile().intrinsics.ppx,
                     color_frame.profile.as_video_stream_profile().intrinsics.ppy,
                     color_frame.profile.as_video_stream_profile().intrinsics.fx,
                     color_frame.profile.as_video_stream_profile().intrinsics.fy]

        return color_image, depth_image, intrinsic
