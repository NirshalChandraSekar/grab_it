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
    

class InferenceMultiCamera:
    def __init__(self):
        self.pipelines = []
        self.aligns = []
        self.device_serials = []
        self.ctx = rs.context()
        
        # Check for connected devices
        if len(self.ctx.devices) == 0:
            raise RuntimeError("No RealSense devices found")
        
        else:
            print("Found", len(self.ctx.devices), "RealSense devices")
        
        # Setup a pipeline for each connected device
        for dev in self.ctx.devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            self.device_serials.append(serial)
            
            pipeline = rs.pipeline()
            config = rs.config()
            # Specify the device using its serial number
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
            pipeline.start(config)
            self.pipelines.append(pipeline)
            
            # Create an align object for aligning depth to color for this device
            self.aligns.append(rs.align(rs.stream.color))
    
    def get_frames(self):
        frames_dict = {}
        
        # Iterate over all pipelines/devices
        for idx, pipeline in enumerate(self.pipelines):
            align = self.aligns[idx]
            serial = self.device_serials[idx]
            
            # Wait for frames from the current pipeline
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Retrieve intrinsics from the color stream
            intrinsics_obj = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = {
                "ppx": intrinsics_obj.ppx,
                "ppy": intrinsics_obj.ppy,
                "fx": intrinsics_obj.fx,
                "fy": intrinsics_obj.fy,
                "width": intrinsics_obj.width,
                "height": intrinsics_obj.height,
                "model": intrinsics_obj.model,
                "coeffs": intrinsics_obj.coeffs
            }
            
            # Store the data in the dictionary with the serial number as the key
            frames_dict[serial] = {
                "color": color_image,
                "depth": depth_image,
                "intrinsics": intrinsics
            }
        
        return frames_dict

    def stop(self):
        # Stop all pipelines when done
        for pipeline in self.pipelines:
            pipeline.stop()