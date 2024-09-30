import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VecKM import NormalEstimator
from VecKM import get_device
from VecKM import to_device


def get_filenames(path):
    with open(path, 'r') as f:
        filenames = f.readlines()
    filenames = [x.strip() for x in filenames]
    return filenames

def load_data(filename):
    """Function to load point clouds and surface normals from files"""
    
    file_path = os.path.join('PCPNet', filename)
    # Load point cloud
    pts = np.loadtxt(file_path + '.xyz')
    # Load surface normals
    normals = np.loadtxt(file_path + '.normals')
    return pts, normals


def find_neighbors(pts, point, neighborhood_radius):
    """Function to find neighbors within a neighborhood radius of a point"""
    neighbors = []
    for pt in pts:
        distance = np.linalg.norm(pt - point)
        if distance <= neighborhood_radius:
            neighbors.append(pt)
    return np.array(neighbors)


def surface_reconstruction(neighbors):
    """Function to reconstruct the surface normal of a point using its neighbors"""
    ground_truth_normal = np.mean(neighbors, axis=0)
    return ground_truth_normal / np.linalg.norm(ground_truth_normal)

def visualize_single_point(pts, normal, pred_normal):
    """Function to visualize point clouds and surface normals"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='k', label='Point Cloud')

    # Plot the ground truth normal
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], color='b', length=0.1, normalize=True, label='Ground Truth Normal')

    # Plot the predicted normal
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], pred_normal[:, 0], pred_normal[:, 1], pred_normal[:, 2], color='r', length=0.1, normalize=True, label='Predicted Normal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Surface Normal Comparison for Single Point')
    plt.show()

def validate_normals(model_path, test_files_list):
    """Function to validate surface normals using the trained model"""
    device = get_device()
    model = to_device(NormalEstimator(), device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for filename in test_files_list:
        test_pts, surface_normals = load_data(filename)
        test_pts_tensor = to_device(torch.from_numpy(test_pts).float(), device)

        idx = random.randint(0, len(test_pts) - 1)
        point = test_pts[idx]

        neighborhood_radius = 0.1
        neighbors = find_neighbors(test_pts, point, neighborhood_radius)
        ground_truth_normal = surface_reconstruction(neighbors)

        with torch.no_grad():
            point_tensor = torch.from_numpy(point).float().cuda().unsqueeze(0)
            pred_normal, _ = model(point_tensor, torch.zeros_like(point_tensor))

        visualize_single_point(neighbors, ground_truth_normal[np.newaxis, :], pred_normal.cpu().numpy())


# List of paths to best models for each test set
best_model_paths = [
    'log/best_model_0.pth',
    'log/best_model_1.pth',
    'log/best_model_2.pth',
    'log/best_model_3.pth',
    'log/best_model_4.pth',
    'log/best_model_5.pth'
]

# List of test filenames
test_filenames = [
    'PCPNet/list/testset_no_noise',
    'PCPNet/list/testset_low_noise',
    'PCPNet/list/testset_med_noise',
    'PCPNet/list/testset_high_noise',
    'PCPNet/list/testset_vardensity_gradient',
    'PCPNet/list/testset_vardensity_striped'
]

# Validate normals for each test set
for model_path, test_filename in zip(best_model_paths, test_filenames):
    test_files_list = get_filenames(test_filename + '.txt')
    validate_normals(model_path, test_files_list)