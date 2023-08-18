import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

# Create 25 example trajectories, each of shape (50, 7)
num_trajectories = 25
trajectory_length = 50
num_dimensions = 7

trajectories = np.load('unguided_multimodality_trajs_without_goal_conditioning.npy')

gui =True
timestep = 1/480

client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)
client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
client_id.setTimeStep(timestep)
client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

target = client_id.getDebugVisualizerCamera()[11]
client_id.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-25,
    cameraTargetPosition=target,
)


p.resetSimulation()
client_id.setGravity(0, 0, -9.8)

plane = client_id.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
workspace = client_id.loadURDF(
    "./Assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True
)
client_id.changeDynamics(
    plane,
    -1,
    lateralFriction=1.1,
    restitution=0.5,
    linearDamping=0.5,
    angularDamping=0.5,
)
client_id.changeDynamics(
    workspace,
    -1,
    lateralFriction=1.1,
    restitution=0.5,
    linearDamping=0.5,
    angularDamping=0.5,
)

panda = client_id.loadURDF("franka_panda/panda.urdf", basePosition=(0, 0, 0), useFixedBase=True)
panda_joints = []
for i in range(client_id.getNumJoints(panda)):
    info = client_id.getJointInfo(panda, i)
    joint_id = info[0]
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    # if joint_name == "ee_fixed_joint":
        # panda_ee_id = joint_id
    if joint_type == client_id.JOINT_REVOLUTE:
        panda_joints.append(joint_id)
# client_id.enableJointForceTorqueSensor(panda, panda_ee_id, 1)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Iterate through trajectories and visualize end effector positions
for i, trajectory in enumerate(trajectories):
    x_vals = []
    y_vals = []
    z_vals = []
    for step in range(trajectory_length):
        joint_positions = trajectory[step]
        
        # Set joint positions for visualization
        for j, joint_angle in enumerate(joint_positions):
            client_id.resetJointState(panda, j, joint_angle)
        
        # Get end effector position
        end_effector_pos =  client_id.getLinkState(panda, 6, computeForwardKinematics=1)[0]
        
        x_vals.append(end_effector_pos[0])
        y_vals.append(end_effector_pos[1])
        z_vals.append(end_effector_pos[2])

    # Plot end effector positions for each trajectory
    ax.plot(x_vals, y_vals, z_vals, label=f"Trajectory {i+1}")

# Set labels and title for the 3D plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("End Effector Positions")

# Show the 3D plot and PyBullet visualization
ax.legend()
plt.show()

# Disconnect from PyBullet
p.disconnect()

