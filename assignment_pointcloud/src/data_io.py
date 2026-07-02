import open3d as o3d
import numpy as np

def load_pointcloud(path: str) -> dict:
    """
    Loads a 3D point cloud from a file and extracts its geometry and metadata into a standard dictionary.
    """
    # Use Open3D's Tensor API (o3d.t) for faster I/O and direct access to point attributes
    t_pcd = o3d.t.io.read_point_cloud(path)

    # Convert the Open3D tensor objects to standard NumPy arrays for easier mathematical manipulation
    pc_data = {
        "points": t_pcd.point.positions.numpy(),
        "colors": t_pcd.point.colors.numpy(),
        "semantic labels": t_pcd.point.label.numpy(),
        "instance ids": t_pcd.point.instance.numpy()
    }

    return pc_data

def validate_pointcloud(pc_data: dict):
    """
    Validates the point cloud data by checking for shape consistency and corrupted values,
    and prints a readable summary of the dataset.
    """
    # Establish the ground-truth number of points based on the XYZ coordinate array
    n_points = pc_data["points"].shape[0]

    for key, array in pc_data.items():
        # Ensure no corrupted or missing values (NaNs) exist
        if np.isnan(array).any():
            raise ValueError(f"NaNs found in {key}!")

        # Ensure every metadata array (colors, labels) has exactly one entry per physical point
        if key != "points":
            if array.shape[0] != n_points:
                raise ValueError(f"Shape mismatch in {key}!")

        # Calculate the total number of unique categories for our summary report
        if key == "semantic labels":
            unique_labels = len(np.unique(array))
        if key == "instance ids":
            unique_instances = len(np.unique(array))

    print(f"Points: {n_points:,}")
    print(f"Unique semantic labels: {unique_labels}")
    print(f"Unique instances: {unique_instances}")