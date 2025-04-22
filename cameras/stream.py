import pyrealsense2 as rs
import numpy as np
import cv2
import os


class CheckCameras:
    '''
    Class to connect to all available cameras and display their live feed.
    '''

    def __init__(self):
        self.context = rs.context()
        self.devices = self.context.query_devices()
        self.device_count = len(self.devices)

        if self.device_count == 0:
            print("No RealSense cameras detected.")
            return
        
        print(f"Number of RealSense cameras detected: {self.device_count}")

    def stream_cameras(self):
        pipelines = []
        for i, device in enumerate(self.devices):
            print(f"Initializing Camera {i + 1}:")
            print(f"  Name: {device.get_info(rs.camera_info.name)}")
            print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device.get_info(rs.camera_info.serial_number))
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            pipelines.append(pipeline)

        while True:
            frames = []
            for i, pipeline in enumerate(pipelines):
                frameset = pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    continue

                color_image = cv2.cvtColor(
                    np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
                )
                frames.append((f"Camera {i + 1}", color_image))

            for window_name, frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    check_cameras = CheckCameras()
    check_cameras.stream_cameras()