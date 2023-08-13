import numpy as np
from scipy.spatial import ConvexHull
import torch
# from polyhedron import Polyhedron
# def cuboid_overlap_volume(cuboid1_vertices, cuboid2_vertices):
#     # Compute the convex hulls of each cuboid
#     hull1 = ConvexHull(cuboid1_vertices)
#     hull2 = ConvexHull(cuboid2_vertices)
    
#     # Compute the intersection of the convex hulls
#     intersect_hull = hull1.intersection(hull2)
    
#     # Calculate the volume of the intersection
#     volume = intersect_hull.volume
    
#     return volume
import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_quaternion_from_vertices(vertices):
    # Calculate the centroid of the vertices
    centroid = np.mean(vertices, axis=0)
    
    # Center the vertices around the centroid
    centered_vertices = vertices - centroid
    
    # Calculate the covariance matrix
    covariance_matrix = np.dot(centered_vertices.T, centered_vertices)
    
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Extract the eigenvector corresponding to the largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Create a rotation matrix that aligns the principal axis with the z-axis
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 2] = principal_axis
    
    # Convert the rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    
    return quaternion, rotation_matrix


voxel_size = 0.1

def calculate_intersection_volume(cuboid1_corners, cuboid2_corners, voxel_size):
    # Calculate the bounding box for both cuboids
    min_coords = np.minimum(np.min(cuboid1_corners, axis=0), np.min(cuboid2_corners, axis=0))
    max_coords = np.maximum(np.max(cuboid1_corners, axis=0), np.max(cuboid2_corners, axis=0))
    
    # Calculate the dimensions of the voxel grid
    dimensions = ((max_coords - min_coords) / voxel_size).astype(int) + 1
    
    # Initialize the voxel grid
    voxel_grid = np.zeros(dimensions, dtype=int)
    
    # Assign unique values to each cuboid
    voxel_grid1 = np.zeros(dimensions, dtype=int)
    voxel_grid2 = np.zeros(dimensions, dtype=int)
    
    # Mark voxels that are inside each cuboid
    for corners, grid in [(cuboid1_corners, voxel_grid1), (cuboid2_corners, voxel_grid2)]:
        indices = ((corners - min_coords) / voxel_size).astype(int)
        for idx, val in enumerate(indices):
            grid[tuple(val)] = idx + 1
    
    # Identify overlapping voxels and calculate the intersection volume
    intersection_voxels = np.logical_and(voxel_grid1 != 0, voxel_grid2 != 0)
    intersection_volume = np.sum(intersection_voxels) * (voxel_size ** 3)
    
    return intersection_volume

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

    return volume

# Define the vertices of the two cuboids
# Cuboid 1 vertices
cuboid1_vertices = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [2.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
    [2.0, 0.0, 2.0],
    [0.0, 2.0, 2.0],
    [2.0, 2.0, 2.0]
], dtype=float)

# Cuboid 2 vertices
cuboid2_vertices = np.array([
    [1.0, 1.0, 1.0],
    [3.0, 1.0, 1.0],
    [1.0, 3.0, 1.0],
    [3.0, 3.0, 1.0],
    [1.0, 1.0, 3.0],
    [3.0, 1.0, 3.0],
    [1.0, 3.0, 3.0],
    [3.0, 3.0, 3.0]
], dtype=float)
# Calculate the intersection volume
intersection_volume = cuboid_overlap_volume2(cuboid1_vertices, cuboid2_vertices)
print("Intersection Volume:", intersection_volume)

# Calculate the quaternion representing the rotation
quaternion, rotation_matrix = calculate_quaternion_from_vertices(cuboid1_vertices)
print("Quaternion:", quaternion)
print("R:", rotation_matrix)

# In this code, you'll need to provide the vertices of the two cuboids in the cuboid1_vertices and cuboid2_vertices arrays. Each vertex is a 3D point (x, y, z) representing a corner of the cuboid.

# Keep in mind that this code uses the ConvexHull method from scipy.spatial to compute the intersection volume, assuming the two cuboids are convex. If the cuboids are not convex or have a more complex shape, you might need to use a more advanced technique to calculate the intersection volume accurately. Additionally, this approach might require a higher computational cost for more intricate shapes or smaller unit sizes.

    # x.grad = None



    # volume.backward()

    # with torch.no_grad():
    #     x -= 0.01 * x.grad