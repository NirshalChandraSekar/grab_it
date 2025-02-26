import numpy as np
import rtde_control # type: ignore
import rtde_receive # type: ignore
from gripper import RobotiqGripper

### Robot Global Variables ###
THUNDER_IP = '192.168.0.101'
LIGHTNING_IP = '192.168.0.102'
THUNDER_HOME = [-np.pi , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0]
LIGHTNING_HOME = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]

THUNDER_GRABIT_HOME = [-3.3489118258105677, -1.3200146716884156, 0.5476034323321741, -0.727967695598938, -0.7958329359637659, -3.3762550989734095]
LIGHTNING_GRABIT_HOME = [-2.9078572432147425, -2.0026475391783656, -0.05930602550506592, -2.6062656841673792, 0.580545961856842, 3.38815975189209]

THUNDER_GRABIT_HOME_EEF_POSE = [0.5133792797815522, 0.09761304103680525, 0.7699079248388268, -1.8114476475549843, -1.7956042618132295, -0.8315509536663834]
LIGHTNING_GRABIT_HOME_EEF_POSE = [-0.5019622959075773, 0.10288288422589228, 0.845704241201291, -1.674059653851395, 1.740895410756312, 1.1504085564683173]

X_DISPLACEMENT = 0.016
Y_DISPLACEMENT = 0.710
Z_DISPLACEMENT = 0.005

SPEED = 0.5
ACCELERATION = 0.5
DT = 0.1
LOOKAHEAD_TIME = 0.2
GAIN = 500

class RobotController:
    def __init__(self, arm : str, need_control: bool = False, need_gripper: bool = False):
        self._ip = THUNDER_IP if arm == 'thunder' else LIGHTNING_IP
        self.home = THUNDER_HOME if arm == 'thunder' else LIGHTNING_HOME
        self.gripper = self._init_gripper() if need_gripper else None
        self.reciever = rtde_receive.RTDEReceiveInterface(self._ip)
        self.controller = rtde_control.RTDEControlInterface(self._ip) if need_control else None

    def _init_gripper(self) -> RobotiqGripper:
        gripper = RobotiqGripper()
        gripper.connect(self._ip, 63352)
        gripper.activate()
        gripper.set_enable(True)
        return gripper

    def get_eff_pose(self) -> list[6]:
        return self.reciever.getActualTCPPose()
    
    def get_joint_angles(self) -> list[6]:
        return self.reciever.getActualQ()
    
    def freeDrive(self):
        self.controller.teachMode()
        while True:
            user_input = input("Enter 'DONE' to Exit Free Drive Mode")
            if user_input == "DONE":
                break
        self.controller.endTeachMode()

    def grasp_object(self):
        self.gripper.set(255)

    def release_object(self):
        self.gripper.set(0)

    def go_home(self):
        self.controller.moveJ(self.home, SPEED, ACCELERATION)

if __name__=="__main__":
    robot = RobotController('thunder', False, False)

