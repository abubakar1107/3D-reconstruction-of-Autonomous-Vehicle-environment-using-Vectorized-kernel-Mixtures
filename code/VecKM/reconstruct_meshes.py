import numpy as np
import open3d as o3d
import torch
from VecKM import NormalEstimator, get_device, to_device

# Load point cloud data
points = np.loadtxt(r'town3_lidar_data.xyz')
# points = np.loadtxt(r'PCPNet/star_smooth100k_noise_white_1.00e-02.xyz')
normals_gt = np.loadtxt(r'PCPNet/star_smooth100k_noise_white_1.00e-02.normals')

# Setup device and model1
device = get_device()
model = to_device(NormalEstimator(), device)
model.load_state_dict(torch.load(r'log\best_model_5.pth', map_location=device))
model.eval()

# Define batch size
batch_size = 5000  # Adjust based on your GPU capacity

# Prepare point data for model and process in batches
point_tensor = torch.from_numpy(points).float().to(device)
num_points = point_tensor.shape[0]
predicted_normals = []

for i in range(0, num_points, batch_size):
    end = min(i + batch_size, num_points)
    batch = point_tensor[i:end]
    pred_normal, _ = model(batch, torch.zeros_like(batch))
    predicted_normals.append(pred_normal.detach().cpu())

# Concatenate all batch results
predicted_normals = np.vstack([n.numpy() for n in predicted_normals])

# Create Open3D point cloud with predicted normals
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)
pcd1.normals = o3d.utility.Vector3dVector(predicted_normals)
# pcd1.normals = o3d.utility.Vector3dVector(normals_gt)
# Visualize the point cloud with predicted normals
o3d.visualization.draw_geometries([pcd1], point_show_normal=True, window_name="Predicted Normals Visualization")

# Orient normals and downsample
pcd1.orient_normals_consistent_tangent_plane(k=10)
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)

# Perform Poisson surface reconstruction
mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1, depth=9)
print("Surface reconstructed.")
print("Initial Vertices:", len(mesh1.vertices))
print("Initial Triangles:", len(mesh1.triangles))


densities1 = np.asarray(densities1)
density_threshold = 0.01
mask = densities1 < density_threshold
if np.any(mask):
    mesh1 = mesh1.remove_vertices_by_mask(mask)

# Check if mesh is still valid
if mesh1 is not None and len(mesh1.vertices) > 0:
    print("Trimmed Vertices:", len(mesh1.vertices))
    print("Trimmed Triangles:", len(mesh1.triangles))
    o3d.visualization.draw_geometries([mesh1], mesh_show_wireframe=True, mesh_show_back_face=True, window_name="Mesh Visualization")
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh1)
    print("Mesh saved successfully.")
else:
    print("Mesh is empty after trimming.")
