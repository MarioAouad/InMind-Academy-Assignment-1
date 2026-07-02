import numpy as np

def scale(pc, factor: float):
    """
    Shrinks or expands the point cloud in-place.
    """
    pc["points"] = pc["points"] * factor
    
def translate(pc, offset: np.ndarray):
    """
    Shifts the entire point cloud in 3D space by an [X, Y, Z] offset vector.
    """
    pc["points"] = pc["points"] + offset
    
def center(pc):
    """
    Moves the point cloud so its physical center (centroid) sits exactly at the origin (0, 0, 0).
    """
    # Calculate the mean across axis=0 to find the center of mass for X, Y, and Z
    centroid = np.mean(pc["points"], axis=0)
    
    # Shift the entire cloud by subtracting the centroid coordinates
    pc["points"] = pc["points"] - centroid
    
def crop_axis_aligned(pc, min_bound, max_bound):
    """
    Crops the point cloud to an axis-aligned bounding box in-place
    Points outside the min_bound and max_bound are removed across all data arrays
    """
    # Generate boolean arrays determining if points are within the X, Y, and Z boundaries
    inside_min = pc["points"] >= min_bound
    inside_max = pc["points"] <= max_bound
    
    valid_coords = inside_min & inside_max
    mask = np.all(valid_coords, axis=1)

    for key in pc.keys():
        pc[key] = pc[key][mask]

def voxel_downsample(pc, voxel_size: float):
    """
    Filters the point cloud by keeping only one point per 3D voxel grid in-place
    """ 
    # Discretize the continuous 3D space into a grid of uniform voxel bins using floor division
    voxel_coords = np.floor(pc["points"] / voxel_size)
    
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    
    for key in pc.keys():
        pc[key] = pc[key][unique_indices]
    
def uniform_downsample(pc, every_k_points: int):
    """Keeps only every k-th point and deletes the rest in-place."""
    for key in pc.keys():
        pc[key] = pc[key][::every_k_points]
    