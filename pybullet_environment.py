import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
import time
import re
import os

class RobotEnvironment:

    def __init__(self, gui = True, timestep = 1/480, manipulator = True):

        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)            # Initialize the bullet client
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())     # Add pybullet's data package to path
        self.client_id.setTimeStep(timestep)                                    # Set simulation timestep
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)        # Disable Shadows

        p.setAdditionalSearchPath(pybullet_data.getDataPath())      # Add pybullet's data package to path   

        self.colors = {'blue': np.array([78, 121, 167]) / 255.0,  # blue
                       'green': np.array([89, 161, 79]) / 255.0,  # green
                       'brown': np.array([156, 117, 95]) / 255.0,  # brown
                       'orange': np.array([242, 142, 43]) / 255.0,  # orange
                       'yellow': np.array([237, 201, 72]) / 255.0,  # yellow
                       'gray': np.array([186, 176, 172]) / 255.0,  # gray
                       'red': np.array([255, 87, 89]) / 255.0,  # red
                       'purple': np.array([176, 122, 161]) / 255.0,  # purple
                       'cyan': np.array([118, 183, 178]) / 255.0,  # cyan
                       'pink': np.array([255, 157, 167]) / 255.0}  # pink
        
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

        if manipulator:
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
        
        links_folder_path = '/home/failedmesh/miniconda3/lib/python3.10/site-packages/pybullet_data/franka_panda/meshes/collision/'
        try:
            link_file_names = os.listdir(links_folder_path)
        except OSError as e:
            print(f"Error reading files in folder: {e}")
       
        self.link_meshes = {}
        self.link_dimensions = {}
        self.link_centers = {}

        self.link_index_to_name = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                                   'hand', 'finger', 'finger', 'finger', 'finger'] 

        for file_name in link_file_names:

            if file_name[-4:] == ".obj":
                vertices = []    
                link_name = file_name[:-4]
                with open(links_folder_path + file_name, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('v '):
                            vertex = re.split(r'\s+', line)[1:4]
                            vertex = np.array([float(coord) for coord in vertex])
                            vertices.append(vertex)
                self.link_meshes[link_name] = np.array(vertices)
                max_point = np.max(self.link_meshes[link_name], axis = 0)
                min_point = np.min(self.link_meshes[link_name], axis = 0)
                self.link_dimensions[link_name] = max_point - min_point
                self.link_centers[link_name] = self.link_dimensions[link_name]/2 + min_point

    def draw_link_bounding_boxes(self):

        self.link_poses = []
        self.link_bounding_vertices = []
        
        self.link_bounding_objs = []

        for link_index in range(0, 11):     # The 12th link (i.e. 11th index) is the grasp target and is not needed   
                
            if link_index not in [8, 9]:    

                link_name = self.link_index_to_name[link_index+1]

                l, b, h = self.link_dimensions[link_name][0], self.link_dimensions[link_name][1], self.link_dimensions[link_name][2]

                if link_index == 7:
                    frame_pos, _ = self.client_id.getLinkState(self.manipulator, 7)[4:6]
                    _, frame_ori = self.client_id.getLinkState(self.manipulator, 10)[4:6]
                elif link_index != -1:
                    frame_pos, frame_ori = self.client_id.getLinkState(self.manipulator, link_index)[4:6]
                else:
                    frame_pos, frame_ori = self.client_id.getBasePositionAndOrientation(self.manipulator)
                world_transform = self.pose_to_transformation(np.array([*frame_pos, *frame_ori]))

                link_dimensions = self.link_dimensions[link_name].copy()
                if link_index == 10:
                    link_dimensions[1] *= 4

                world_link_center = (world_transform @ np.vstack((np.expand_dims(self.link_centers[link_name], 1), 1)))[:-1, 0]

                vertices = np.array([[-l/2, -b/2, -h/2],
                                    [ l/2, -b/2, -h/2],
                                    [ l/2,  b/2, -h/2],
                                    [-l/2,  b/2, -h/2],
                                    [-l/2, -b/2,  h/2],
                                    [ l/2, -b/2,  h/2],
                                    [ l/2,  b/2,  h/2],
                                    [-l/2,  b/2,  h/2]])
                vertices = vertices + np.array([self.link_centers[link_name]])
                vertices = world_transform @ np.vstack((vertices.T, np.ones(8)))
                self.link_bounding_vertices.append(vertices.T[:, :-1])

                self.link_poses.append(np.array([*world_link_center, *frame_ori]))
                
                vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                        halfExtents = link_dimensions/2,
                                        rgbaColor = np.hstack([self.colors['red'], np.array([1.0])]))
                
                obj_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                        basePosition = world_link_center, 
                                                        baseOrientation = frame_ori)
                
                self.link_bounding_objs.append(obj_id)
    
    def clear_bounding_boxes(self):

        for obj_id in self.link_bounding_objs:
            self.client_id.removeBody(obj_id)
    
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

    def draw_frame(self, transform, scale_factor = 0.2):

        unit_axes_world = np.array([[scale_factor, 0, 0], 
                                    [0, scale_factor, 0], 
                                    [0, 0, scale_factor],
                                    [1, 1, 1]])
        axis_points = ((transform @ unit_axes_world)[:3, :]).T
        axis_center = transform[:3, 3]

        self.client_id.addUserDebugLine(axis_center, axis_points[0], self.colors['red'], lineWidth = 4)
        self.client_id.addUserDebugLine(axis_center, axis_points[1], self.colors['green'], lineWidth = 4)
        self.client_id.addUserDebugLine(axis_center, axis_points[2], self.colors['blue'], lineWidth = 4)
        
    
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

        rotation_matrix = self.quaternion_to_rotation_matrix(quat)

        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_matrix.copy()
        transform[:3, 3] = pos.copy()
        transform[3, 3] = 1

        return transform

    def euler_to_rotation_matrix(yaw, pitch, roll):
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

        R = Rz @ (Ry @ Rx)
        
        return R
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.
        
        :param q: Quaternion [w, x, y, z]
        :return: 3x3 rotation matrix
        """
        # w, x, y, z = quat
        # rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,        2*x*z + 2*y*w],
        #                             [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        #                             [2*x*z - 2*y*w,        2*y*z + 2*x*w,        1 - 2*x**2 - 2*y**2]])
        
        mat = np.array(self.client_id.getMatrixFromQuaternion(quat))
        rotation_matrix = np.reshape(mat, (3, 3))

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

        





