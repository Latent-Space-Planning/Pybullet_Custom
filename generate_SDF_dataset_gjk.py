
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
from time import sleep
import sys
import time
import numpy as np
import torch
import os
from diffusion.Models.temporalunet import TemporalUNet
from diffusion.infer_diffusion import infer, infer_guided, value_func_and_grad
from train_SDF import SDFModel
from gjk_trajopt_pybullet import RobotEnvironment

sys.path.append('/home/jayaram/research/research_tracks/table_top_rearragement/Pybullet_Custom/openGJK/examples/cython')
import openGJK_cython as opengjk

lower_lims = np.array([-2.8973000049591064, -1.7627999782562256, -2.8973000049591064, -3.0717999935150146, -2.8973000049591064, -0.017500000074505806, -2.8973000049591064])
upper_lims = np.array([2.8973000049591064, 1.7627999782562256, 2.8973000049591064, -0.0697999969124794, 2.8973000049591064, 3.752500057220459, 2.8973000049591064])

# gui =True
# timestep = 1/480


# client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)
# client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
# client_id.setTimeStep(timestep)
# client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())



# target = client_id.getDebugVisualizerCamera()[11]
# client_id.resetDebugVisualizerCamera(
#     cameraDistance=1.5,
#     cameraYaw=90,
#     cameraPitch=-25,
#     cameraTargetPosition=target,
# )


# p.resetSimulation()
# client_id.setGravity(0, 0, -9.8)

# plane = client_id.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
# workspace = client_id.loadURDF(
#     "./Assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True
# )
# client_id.changeDynamics(
#     plane,
#     -1,
#     lateralFriction=1.1,
#     restitution=0.5,
#     linearDamping=0.5,
#     angularDamping=0.5,
# )
# client_id.changeDynamics(
#     workspace,
#     -1,
#     lateralFriction=1.1,
#     restitution=0.5,
#     linearDamping=0.5,
#     angularDamping=0.5,
# )

# panda = client_id.loadURDF("franka_panda/panda.urdf", basePosition=(0, 0, 0), useFixedBase=True)
# panda_joints = []
# for i in range(client_id.getNumJoints(panda)):
#     info = client_id.getJointInfo(panda, i)
#     joint_id = info[0]
#     joint_name = info[1].decode("utf-8")
#     joint_type = info[2]
#     # if joint_name == "ee_fixed_joint":
#         # panda_ee_id = joint_id
#     if joint_type == client_id.JOINT_REVOLUTE:
#         panda_joints.append(joint_id)
# # client_id.enableJointForceTorqueSensor(panda, panda_ee_id, 1)
import math

def calculate_cuboid_corners(cuboid_center, cuboid_dimensions):
    half_dims = [dim * 0.5 for dim in cuboid_dimensions]

    corners = [
        [cuboid_center[0] + half_dims[0], cuboid_center[1] + half_dims[1], cuboid_center[2] + half_dims[2]],
        [cuboid_center[0] + half_dims[0], cuboid_center[1] + half_dims[1], cuboid_center[2] - half_dims[2]],
        [cuboid_center[0] + half_dims[0], cuboid_center[1] - half_dims[1], cuboid_center[2] + half_dims[2]],
        [cuboid_center[0] + half_dims[0], cuboid_center[1] - half_dims[1], cuboid_center[2] - half_dims[2]],
        [cuboid_center[0] - half_dims[0], cuboid_center[1] + half_dims[1], cuboid_center[2] + half_dims[2]],
        [cuboid_center[0] - half_dims[0], cuboid_center[1] + half_dims[1], cuboid_center[2] - half_dims[2]],
        [cuboid_center[0] - half_dims[0], cuboid_center[1] - half_dims[1], cuboid_center[2] + half_dims[2]],
        [cuboid_center[0] - half_dims[0], cuboid_center[1] - half_dims[1], cuboid_center[2] - half_dims[2]]
    ]

    return np.array(corners)

def find_cuboid_bbox_parameters_minimal(sphere_center, sphere_radius):
    # Calculate cuboid dimensions (diagonal <= 2 * sphere_radius)
    bbox_params = np.zeros((8, 3), dtype=float)  
    bbox_dims = np.zeros(3, dtype=float) 
    bbox_centres = np.zeros(3, dtype=float) 

    cuboid_dims = [sphere_radius * 2 ] * 3
    bbox_dims = np.array(cuboid_dims)
    bbox_centres = np.array(sphere_center)

    bbox_params = calculate_cuboid_corners(sphere_center, cuboid_dims)

    return bbox_params, bbox_centres, bbox_dims  

def find_cuboid_bbox_parameters(sphere_center, sphere_radius):
    # Calculate cuboid dimensions (diagonal <= 2 * sphere_radius)
    bbox_params = np.zeros((7, 8, 3), dtype=float)  
    bbox_dims = np.zeros((7, 3), dtype=float) 
    bbox_centres = np.zeros((7, 3), dtype=float) 

    main_cuboid_dims = [sphere_radius * 2 / math.sqrt(3)] * 3
    bbox_dims[0] = np.array(main_cuboid_dims)
    main_cuboid_side = main_cuboid_dims[0]
    bbox_centres[0] = np.array(sphere_center)

    bbox_params[0] = calculate_cuboid_corners(sphere_center, main_cuboid_dims)

    filling_cuboid_dims = {}
    filling_cuboid_dims[1] = [sphere_radius - main_cuboid_side/2.0, main_cuboid_side, main_cuboid_side]
    filling_cuboid_dims[2] = [sphere_radius - main_cuboid_side/2.0, main_cuboid_side, main_cuboid_side]

    filling_cuboid_dims[3] = [main_cuboid_side, sphere_radius - main_cuboid_side/2.0, main_cuboid_side]
    filling_cuboid_dims[4] = [main_cuboid_side, sphere_radius - main_cuboid_side/2.0, main_cuboid_side]

    filling_cuboid_dims[5] = [main_cuboid_side, main_cuboid_side, sphere_radius - main_cuboid_side/2.0]
    filling_cuboid_dims[6] = [main_cuboid_side, main_cuboid_side, sphere_radius - main_cuboid_side/2.0]

    for i in range(6):
        bbox_dims[i + 1] = np.array(filling_cuboid_dims[i + 1])

    offset = sphere_radius/2.0 + main_cuboid_side/4.0

    filling_cuboid_centers = {}
    filling_cuboid_centers[1] = [sphere_center[0] + offset, sphere_center[1], sphere_center[2]]
    filling_cuboid_centers[2] = [sphere_center[0] - offset, sphere_center[1], sphere_center[2]]

    filling_cuboid_centers[3] = [sphere_center[0], sphere_center[1] + offset, sphere_center[2]]
    filling_cuboid_centers[4] = [sphere_center[0], sphere_center[1] - offset, sphere_center[2]]

    filling_cuboid_centers[5] = [sphere_center[0], sphere_center[1], sphere_center[2] + offset]
    filling_cuboid_centers[6] = [sphere_center[0], sphere_center[1], sphere_center[2] - offset]

    for i in range(6):
        bbox_centres[i + 1] = np.array(filling_cuboid_centers[i + 1])

    for i in range(6): 
        bbox_params[i + 1] = calculate_cuboid_corners(filling_cuboid_centers[i + 1], filling_cuboid_dims[i + 1])

    return bbox_params, bbox_centres, bbox_dims  

def generate_collision_course4(client_id):
    obstacle_centers = []
    sphereRadius = 0.08
    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.6, -0.10, 0.7])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.58, -0.09, 0.3])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    colVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 1, 0, 1])
    # colSphereId = client_id.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    obstacle_center = np.array([0.4, -0.16, 0.4])
    obstacle_centers.append(obstacle_center)
    colSph = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=colVizshape, basePosition=obstacle_center)

    q0 = np.array([-0.0493874, 1.2605693, 0.36301115, -0.26907736, -0.2328229, 1.1613771, -1.6480199 ]) 
    qTarget = np.array([-1.5038756, 0.9193187, 1.8278371, -1.4488376, -1.1391141, 1.2231112, -0.9670128])

    st_pos = np.array([0.75024617,  0.01952344,  0.32572699])
    end_pos = np.array([0.53903937, -0.27098262,  0.58423716])

    return obstacle_centers, q0, qTarget, st_pos, end_pos

def move_joints(client_id, panda, panda_joints, target_joints, speed=0.01, timeout=3):
    """Move UR5e to target joint configuration."""
    t0 = time.time()
    all_joints = []
    while (time.time() - t0) < timeout:
        current_joints = np.array([client_id.getJointState(panda, i)[0] for i in panda_joints])
        # pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
        # if pos[2] < 0.005:
        #     print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
        #     return False, []
        diff_joints = target_joints - current_joints
        if all(np.abs(diff_joints) < 1e-2):
            # give time to stop
            for _ in range(10):
                client_id.stepSimulation()
            return True, all_joints

        # Move with constant velocity
        norm = np.linalg.norm(diff_joints)
        v = diff_joints / norm if norm > 0 else 0
        step_joints = current_joints + v * speed
        all_joints.append(step_joints)
        client_id.setJointMotorControlArray(
            bodyIndex=panda,
            jointIndices=panda_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=step_joints,
            positionGains=np.ones(len(panda_joints)),
        )
        client_id.stepSimulation()
    print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
    return False, []

def cuboid_overlap_volume2(cuboid1_vertices, cuboid2_vertices):

    # cub1_rot = calculate_quaternion_from_vertices(cuboid1_vertices)
    # cub2_rot = calculate_quaternion_from_vertices(cuboid2_vertices)

    cuboid1_vertices = torch.from_numpy(cuboid1_vertices).t()
    cuboid2_vertices = torch.from_numpy(cuboid2_vertices).t()

    x_min = torch.min(cuboid1_vertices, dim = 1)[0]
    x_max = torch.max(cuboid1_vertices, dim = 1)[0]

    a_min = torch.min(cuboid2_vertices, dim = 1)[0]
    a_max = torch.max(cuboid2_vertices, dim = 1)[0]

    overlap_min = torch.max(x_min, a_min)
    # print(overlap_min)
    overlap_max = torch.min(x_max, a_max)
    # print(overlap_max)

    overlap_lengths = overlap_max - overlap_min
    # print(overlap_lengths)
    volume = torch.prod(torch.clamp(overlap_lengths, min = 0))

    return volume.item()

# obstacle_centers, q0, qTarget, st_pos, end_pos = generate_collision_course4(client_id=client_id)

# sphereRadius = 0.05
# inVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
# inViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=inVizshape, basePosition=st_pos) # np.array([0.4372005 , -0.14387664,  0.5400902]))

# sphereRadius = 0.05
# outVizshape = client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 1, 1])
# outViz = client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=outVizshape, basePosition=end_pos) #  np.array([0.2937204 ,  0.10894514,  0.5569069]))

# Inferring from Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
traj_len=50
T = 255
model_name = "/home/jayaram/research/research_tracks/table_top_rearragement/global_classifier_guidance_for_7DOF_manipulator/diffuser_ckpts_7dof_mpinets/7dof/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
if not os.path.exists(model_name):
    print("Model does not exist for these parameters. Train a model first.")
    _ = input("Press anything to exit")
    exit()
denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = 32, dims=(32, 64, 128, 256, 512, 512))
_ = denoiser.to(device)

# q0 = np.array([ 0.9798614,  0.5573511, -0.1629493, -1.9912002, -0.23168434, 2.214461, -0.38091224])
# qTarget = np.array([2.2053013 ,  1.2142502 , -2.5370142 , -2.4402852 ,  1.0121354 ,1.2068753 , -1.7239444]) # 
# getLinkStates(panda, )

print("Robot Information!!!")

# def getMotorJointStates(robot):
#     joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
#     joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
#     joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
#     joint_positions = [state[0] for state in joint_states]
#     joint_velocities = [state[1] for state in joint_states]
#     joint_torques = [state[3] for state in joint_states]
#     return joint_positions, joint_velocities, joint_torques

# print(getMotorJointStates(panda))


print("Robot info ended!!!!")

# sleep(10)

st_time = time.time()
# trajectory = infer(denoiser, q0, qTarget)
# trajectory = infer_guided(denoiser, q0, qTarget, obstacle_centers, client_id, panda, panda_joints)  #(50, 7)
trajectory = np.zeros((50, 7))
end_time = time.time()
print("-"*10)
print(f"Total Inference Time: {end_time - st_time} seconds")
print("-"*10)
print('traj shape: {}'.format(trajectory.shape))

# for l, joint_ind in enumerate(panda_joints):
#     client_id.resetJointState(panda, joint_ind, q0[l])

num_samples = 50000  # You can change this to generate a different number of samples
trajectory = np.random.uniform(lower_lims, upper_lims, size=(num_samples, len(lower_lims)))  #(50000, 7)
print('sdf joint angle dataset shape: {}'.format(trajectory.shape))
file_name = "joint_states.npy"
np.save(file_name, trajectory)

env = RobotEnvironment()
obstacle_centers, q0, qTarget, st_pos, end_pos = env.generate_collision_course_gjk()

sphereRadius = 0.05
inVizshape = env.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
inViz = env.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=inVizshape, basePosition=st_pos) # np.array([0.4372005 , -0.14387664,  0.5400902]))

sphereRadius = 0.05
outVizshape = env.client_id.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[1, 0, 1, 1])
outViz = env.client_id.createMultiBody(baseMass=0, baseVisualShapeIndex=outVizshape, basePosition=end_pos) #  np.array([0.2937204 ,  0.10894514,  0.5569069]))

panda = env.manipulator
panda_joints = env.joints  #[0, 1, 2, 3, 4, 5, 6] -- only revolute joints

# generate bboxes for each collision object in env (considering spheres only for now)
# obstacles_bbox_vertices =  np.zeros((len(obstacle_centers), 7, 8, 3), dtype=float)  
# obstacles_bbox_centers =  np.zeros((len(obstacle_centers), 7, 3), dtype=float)  
# obstacles_bbox_dims =  np.zeros((len(obstacle_centers), 7, 3), dtype=float) 
obstacles_bbox_vertices =  np.zeros((len(obstacle_centers), 8, 3), dtype=float)  
obstacles_bbox_centers =  np.zeros((len(obstacle_centers), 3), dtype=float)  
obstacles_bbox_dims =  np.zeros((len(obstacle_centers), 3), dtype=float)  
for i in range(len(obstacle_centers)):
    (obstacles_bbox_vertices[i], obstacles_bbox_centers[i], obstacles_bbox_dims[i]) = find_cuboid_bbox_parameters_minimal(obstacle_centers[i], 0.06)

print('obstacles bbox vertices shape: {}'.format(obstacles_bbox_vertices.shape)) #(3, 8, 3)  #(3, 7, 8, 3)
file_name = "obstacles_bbox_vertices.npy"
np.save(file_name, obstacles_bbox_vertices)
env.draw_obstacle_bounding_boxes(obstacles_bbox_vertices, obstacles_bbox_centers, obstacles_bbox_dims)

num_links = 10  #2 prismatic joints (fingers) are merged into 1 bbox and grasp target link is not considered  -- (12-2)
link_vertices =  np.zeros((trajectory.shape[0], num_links, 8, 3), dtype=float) #(50k, 10, 8, 3)

# Get bbox of all links for all conf
dists =  np.zeros((trajectory.shape[0], num_links, len(obstacle_centers)), dtype=float) 
for conf_no in range(len(trajectory)):    #iterate thru all conf
    for l, joint_ind in enumerate(panda_joints):   #reset joints according to current conf 
        env.client_id.resetJointState(panda, joint_ind, trajectory[conf_no, l])

    numJoints = p.getNumJoints(panda)  #12 in total
    env.draw_link_bounding_boxes()    #required to get each link bbox vertices

    link_vertices[conf_no] = np.array(env.link_bounding_vertices)
    # print('**************')

print('link bbox vertices shape: {}'.format(link_vertices.shape))  #(50k, 10, 8, 3)
file_name = "links_bbox_vertices.npy"
np.save(file_name, link_vertices)

# 1: Create data of each link from every obstacle: 
dists =  np.zeros((trajectory.shape[0], num_links, len(obstacle_centers)), dtype=float) #(50k, 10, 3)
for conf_no in range(len(trajectory)):    #iterate thru all conf
    numJoints = p.getNumJoints(panda)  #12 in total

    for link_no in range(num_links):   #iterate thru all links
        for obstacle_no in range(len(obstacle_centers)):    #iterate thru all obstacles

            val = 100.0 * opengjk.pygjk(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
            # print(value)
            if(val > 0.0): 
                dists[conf_no, link_no, obstacle_no] = val
            else: 
                # print('dist is 0.0, so compute volume')
                dists[conf_no, link_no, obstacle_no] = -100000.0 * cuboid_overlap_volume2(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
                # print(-100000.0 * cuboid_overlap_volume2(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no]))
    # print('**************')

file_name = "sdf_distances_all.npy"
np.save(file_name, dists)

# 2: Create data of min distance of each link: 
dists =  np.zeros((trajectory.shape[0], num_links), dtype=float) #(50k, 10)
for conf_no in range(len(trajectory)):    #iterate thru all conf
    numJoints = p.getNumJoints(panda)  #12 in total

    for link_no in range(num_links):   #iterate thru all links
        min_dist_of_link_from_obs = np.inf
        for obstacle_no in range(len(obstacle_centers)):
            val = 100.0 * opengjk.pygjk(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
            if(val > 0.0): 
                min_dist_of_link_from_obs = min(val, min_dist_of_link_from_obs)
            else:
                val = -100000.0 * cuboid_overlap_volume2(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
                min_dist_of_link_from_obs = min(val, min_dist_of_link_from_obs)
        print('min_dist_of_link_from_obs: {}'.format(min_dist_of_link_from_obs))
        dists[conf_no, link_no] = min_dist_of_link_from_obs
    # print('**************')

file_name = "sdf_distances_per_link.npy"
np.save(file_name, dists)

# 3: Create classification data: 
labels =  np.zeros(trajectory.shape[0], dtype=float) #(50k, )   # 1-- in collision
for conf_no in range(len(trajectory)):    #iterate thru all conf
    numJoints = p.getNumJoints(panda)  #12 in total

    break_out = False
    for link_no in range(num_links):   #iterate thru all links
        for obstacle_no in range(len(obstacle_centers)):    #iterate thru all obstacles
            val = opengjk.pygjk(link_vertices[conf_no, link_no], obstacles_bbox_vertices[obstacle_no])
            # print(value)
            if(val < 0.0 ): 
                labels[conf_no] = 1.0 
                break_out = True
                break

        if(break_out): 
            break_out = False
            break
    # print('**************')

file_name = "collision data.npy"
np.save(file_name, labels)

# print('dits shape: {}'.format(dists.shape))
# file_name = "sdf_dists_three_spheres.npy"

# # Save the 3D numpy array to the specified file
# np.save(file_name, dists)

# train sdf on distance to obstacles data


# sleep(5)
# for i in range(len(trajectory)):
#     sleep(0.4)
#     current_joints = np.array([env.client_id.getJointState(panda, i)[0] for i in panda_joints])
#     target_joints = trajectory[i]
#     print(f"Current Joints: {current_joints}")
#     print(f"Target Joints: {target_joints}")
#     print(f"Itr number: {i}")
#     move_joints(env.client_id, panda, panda_joints, target_joints)
#     env.client_id.stepSimulation()

# sleep(100)