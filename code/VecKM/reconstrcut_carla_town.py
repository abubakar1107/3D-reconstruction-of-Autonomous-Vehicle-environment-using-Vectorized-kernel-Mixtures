import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from VecKM import NormalEstimator, get_device, to_device

# Load point cloud data and normals
points = np.loadtxt(r'town3_lidar_data.xyz')
# points = np.loadtxt(r'PCPNet/sphere100k_noise_white_1.00e-02.xyz')
normals_gt = np.loadtxt(r'PCPNet/sphere100k_noise_white_1.00e-02.normals')
# Setup device and model
device = get_device()
model = to_device(NormalEstimator(), device)
model.load_state_dict(torch.load(r'log/best_model_5.pth', map_location=device))
model.eval()

batch_size = 10000  # Adjust this based on your GPU memory capacity
pred_normal = []

for i in range(0, len(points), batch_size):
    point_batch = points[i:i+batch_size]
    point_tensor = torch.from_numpy(point_batch).float().to(device)
    
    with torch.no_grad():
        batch_pred_normal, _ = model(point_tensor, torch.zeros_like(point_tensor))
        pred_normal.append(batch_pred_normal.detach().cpu().numpy())

# Concatenate all the predicted normals
pred_normal = np.concatenate(pred_normal, axis=0)











# Create Open3D point cloud with predicted normals
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)

pcd1.normals = o3d.utility.Vector3dVector(pred_normal)
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

# Optionally, trim the mesh based on densities
densities1 = np.asarray(densities1)
density_threshold = 0.01
mask = densities1 < density_threshold
if np.any(mask):
    mesh1 = mesh1.remove_vertices_by_mask(mask)

# Check if mesh is still valid
if mesh1 is not None and len(mesh1.vertices) > 0:
    print("Trimmed Vertices:", len(mesh1.vertices))
    print("Trimmed Triangles:", len(mesh1.triangles))
    
    # Colorize the mesh using densities or normals
    densities1 = (densities1 - densities1.min()) / (densities1.max() - densities1.min())  # Normalize densities to [0, 1]
    colors = plt.cm.viridis(densities1)[:, :3]  # Use a colormap to get colors
    mesh1.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh1], mesh_show_wireframe=True, mesh_show_back_face=True, window_name="Colorful Mesh Visualization")
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh1)
    print("Mesh saved successfully.")
else:
    print("Mesh is empty after trimming.")