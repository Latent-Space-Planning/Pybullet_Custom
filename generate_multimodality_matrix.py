import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

trajectories = np.load('unguided_multimodality_trajs_without_goal_conditioning.npy')

# Reshape trajectories to (25, 350) for cosine similarity calculation
reshaped_trajectories = [trajectory.reshape(-1) for trajectory in trajectories]
stacked_trajectories = np.vstack(reshaped_trajectories)

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(stacked_trajectories, stacked_trajectories)

# Create a heatmap using seaborn
plt.figure(figsize=(18, 10))
sns.heatmap(cosine_sim_matrix, cmap='coolwarm', annot=True, fmt=".3f", xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Heatmap')
plt.xlabel('Trajectory Index')
plt.ylabel('Trajectory Index')
plt.tight_layout()
plt.show()