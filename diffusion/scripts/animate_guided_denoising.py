import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from scripts.guide_functions import *

# -------------- Edit these parameters -------------- #

env_name = "two_pillars"

# --------------------------------------------------- #

# Create a figure and axis to plot on
fig, ax = plt.subplots()

diffusion_trajs = np.load("reverse_guided_trajectory_" + env_name + ".npy")

env_path = "environments/2D/" + env_name + "/"
env = np.load(env_path + "env.npy")

pixel_traj = point_to_pixel(diffusion_trajs, env.shape)
pixel_traj[:, 1, :] = env.shape[1] - 1 - pixel_traj[:, 1, :]

T = diffusion_trajs.shape[0]

# Define the animation function
def animate(i):
    # Update the y data with a new sine wave shifted in the x direction
    ax.clear()

    ax.imshow(np.rot90(env), cmap = 'gray')

    ax.scatter(pixel_traj[T-1-i, 0, 1:-1], pixel_traj[T-1-i, 1, 1:-1])
    ax.scatter(pixel_traj[T-1-i, 0, 0], pixel_traj[T-1-i, 1, 0], color = 'green')
    ax.scatter(pixel_traj[T-1-i, 0, -1], pixel_traj[T-1-i, 1, -1], color = 'red')

    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])

    plt.title("Time step = " + str(T-1-i))

    os.system("cls")
    print("Animated ", i, " time steps")

# Create the animation object, with a 50ms delay between frames
ani = FuncAnimation(fig, animate, frames = T, interval=1000, repeat=False)

# Set up the writer to save the animation as an MP4 video
writer = FFMpegWriter(fps=60)

# Save the animation as an MP4 video file
ani.save("reverse_guided_diffusion_" + env_name + ".mp4", writer=writer)

# Display the animation
#plt.show()

