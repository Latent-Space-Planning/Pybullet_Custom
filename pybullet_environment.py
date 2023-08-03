import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
import time

class RobotEnvironment:

    def __init__(self, gui = True, timestep = 1/480):

        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)            # Initialize the bullet client
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())     # Add pybullet's data package to path
        self.client_id.setTimeStep(timestep)                                    # Set simulation timestep
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)        # Disable Shadows

        p.setAdditionalSearchPath(pybullet_data.getDataPath())      # Add pybullet's data package to path   

        target = self.client_id.getDebugVisualizerCamera()[11]          # Get cartesian coordinates of the camera's focus
        self.client_id.resetDebugVisualizerCamera(                      # Reset initial camera position
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=target,
        )
        
        p.resetSimulation()                             
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

        self.plane = self.client_id.loadURDF("plane.urdf", basePosition=(0, 0, 0), useFixedBase=True)   # Load a floor
        
        self.client_id.changeDynamics(                  # Set physical properties of the floor
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        self.initialize_manipulator()

    def initialize_manipulator(self, urdf_file = "franka_panda/panda.urdf", base_position = (0, 0, 0)):

        self.manipulator = self.client_id.loadURDF(urdf_file, basePosition = base_position, useFixedBase = True)
        self.joints = []

        for i in range(self.client_id.getNumJoints(self.manipulator)):

            info = self.client_id.getJointInfo(self.manipulator, i)

            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]

            if joint_name == "panda_grasptarget_hand":
                self.end_effector = joint_id

            if joint_type == self.client_id.JOINT_REVOLUTE:
                self.joints.append(joint_id)

        self.joint_lower_limits = np.array([-166*(np.pi/180), 
                                           -101*(np.pi/180), 
                                           -166*(np.pi/180), 
                                           -176*(np.pi/180),
                                           -166*(np.pi/180),
                                           -1*(np.pi/180),
                                           -166*(np.pi/180)])
        
        self.joint_upper_limits = np.array([166*(np.pi/180), 
                                           101*(np.pi/180), 
                                           166*(np.pi/180), 
                                           -4*(np.pi/180),
                                           166*(np.pi/180),
                                           215*(np.pi/180),
                                           166*(np.pi/180)])

    def get_joint_positions(self):

        return np.array([self.client_id.getJointState(self.manipulator, i)[0] for i in self.joints])
    
    def get_joint_velocities(self):

        return np.array([self.client_id.getJointState(self.manipulator, i)[1] for i in self.joints])

    def get_tf_mat(self, i, joint_angles):
        
        dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                    [0, 0, -np.pi / 2, joint_angles[1]],
                    [0, 0.316, np.pi / 2, joint_angles[2]],
                    [0.0825, 0, np.pi / 2, joint_angles[3]],
                    [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                    [0, 0, np.pi / 2, joint_angles[5]],
                    [0.088, 0, np.pi / 2, joint_angles[6]],
                    [0, 0.107, 0, 0],
                    [0, 0, 0, -np.pi / 4],
                    [0.0, 0.1034, 0, 0]], dtype=np.float64)
        
        a = dh_params[i][0]
        d = dh_params[i][1]
        alpha = dh_params[i][2]
        theta = dh_params[i][3]
        q = theta

        return np.array([[np.cos(q), -np.sin(q), 0, a],
                        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                        [0, 0, 0, 1]])

    def forward_kinematics(self, joint_angles):

        T_EE = np.identity(4)
        for i in range(7 + 3):
            T_EE = T_EE @ self.get_tf_mat(i, joint_angles)

        return T_EE
    
    def get_jacobian(self, joint_angles):

        T_EE = self.get_ee_tf_mat(joint_angles)

        J = np.zeros((6, 10))
        T = np.identity(4)
        for i in range(7 + 3):
            T = T @ self.get_tf_mat(i, joint_angles)

            p = T_EE[:3, 3] - T[:3, 3]
            z = T[:3, 2]

            J[:3, i] = np.cross(z, p)
            J[3:, i] = z

        return J[:, :7]
    
    def inverse_kinematics(self, point, euler_orientation = None):

        if type(euler_orientation) == type(None):
            point = self.client_id.calculateInverseKinematics(self.manipulator, self.end_effector, point)
        else:
            quat = self.client_id.getQuaternionFromEuler(euler_orientation)
            point = self.client_id.calculateInverseKinematics(self.manipulator, self.end_effector, point, quat)

        return point
    
    def get_end_effector_pose(self):

        pos, ori = self.client_id.getLinkState(self.manipulator, 11, computeForwardKinematics = 1)[:2]
        pose = np.array([*pos, *ori])

        return pose
    
    def get_end_effector_transformation(self):

        pose = self.get_end_effector_pose()        
        transform = self.pose_to_transformation(pose)

        return transform

    def pose_to_transformation(self, pose):

        pos = pose[:3]
        quat = pose[3:]

        rotation_matrix = self.get_rotation_matrix(quat)

        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_matrix.copy()
        transform[:3, 3] = pos.copy()
        transform[3, 3] = 1

        return transform

    def get_rotation_matrix(self, quat):

        mat = np.array(self.client_id.getMatrixFromQuaternion(quat))
        rotation_matrix = np.reshape(mat, (3, 3))
        #rotation_matrix = np.concatenate([[mat[:3]], [mat[3:6]], [mat[6:9]]], axis = 0)

        return rotation_matrix
        
    def move_joints(self, target_joint_pos, speed = 0.01, timeout = 3):

        t0 = time.time()
        all_joints = []

        while (time.time() - t0) < timeout:
          
            current_joint_pos = self.get_joint_positions()
            error = target_joint_pos - current_joint_pos

            if all(np.abs(error) < 1e-2):
                for _ in range(10):
                    self.client_id.stepSimulation()     # Give time to stop
                return True, all_joints
            
            norm = np.linalg.norm(error)
            vel = error / norm if norm > 0 else 0
            next_joint_pos = current_joint_pos + vel * speed
            all_joints.append(next_joint_pos)

            self.client_id.setJointMotorControlArray(           # Move with constant velocity
                bodyIndex = self.manipulator,
                jointIndices = self.joints,
                controlMode = p.POSITION_CONTROL,
                targetPositions = next_joint_pos,
                positionGains = np.ones(len(self.joints)),
            )

            self.client_id.stepSimulation()

        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")

        return False, all_joints
    
    def go_home(self):

        self.move_joints(np.zeros((7,)))

    def execute_trajectory(self, trajectory):

        lower_limit = np.array([-2.8973000049591064, -1.7627999782562256, -2.8973000049591064, -3.0717999935150146, -2.8973000049591064, -0.017500000074505806, -2.8973000049591064])
        upper_limit = np.array([2.8973000049591064, 1.7627999782562256, 2.8973000049591064, -0.0697999969124794, 2.8973000049591064, 3.752500057220459, 2.8973000049591064])

        for i, joint_ind in enumerate(self.joints):
            self.client_id.resetJointState(self.manipulator, joint_ind, trajectory[0, i])

        _ = input("Press Enter to execute trajectory")

        for i in range(1, trajectory.shape[-1]):
            time.sleep(0.4)
            # current_joints = np.array([self.client_id.getJointState(self.manipulator, i)[0] for i in self.joints])
            target_joints = trajectory[:, i]
            # print(f"Current Joints: {current_joints}")
            # print(f"Target Joints: {target_joints}")
            # print(f"Itr number: {i}")

            if any(target_joints <= lower_limit) or any(target_joints >= upper_limit):

                print("Joint Limits Exceeded")

            self.move_joints(target_joints)
            self.client_id.stepSimulation()

        





