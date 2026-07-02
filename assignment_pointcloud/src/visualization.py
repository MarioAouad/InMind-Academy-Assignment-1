import open3d as o3d
import numpy as np
import copy
import os
import time
from processing import center, crop_axis_aligned, voxel_downsample, uniform_downsample

class PointCloudViewer:
    def __init__(self, pc_data: dict):
        self.original_pc = copy.deepcopy(pc_data)
        self.pc = copy.deepcopy(pc_data)
        self.o3d_pcd = o3d.geometry.PointCloud()
        
        self.color_mode = "RGB"  
        self.voxel_size = 0.5    
        self.uniform_k = 10       # Keeps 1 out of every 10 points
        self.is_downsampled = False
        
        self.update_o3d_geometry()

    def update_o3d_geometry(self):
        self.o3d_pcd.points = o3d.utility.Vector3dVector(self.pc["points"])
        
        if self.color_mode == "RGB":
            colors = get_raw_colors(self.pc)
        elif self.color_mode == "SEMANTIC":
            colors = color_by_semantic_label(self.pc["semantic labels"])
        elif self.color_mode == "INSTANCE":
            colors = color_by_instance_id(self.pc["instance ids"])
            
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="3D Perception Viewer", width=1280, height=720)
        vis.add_geometry(self.o3d_pcd)
        
        vis.register_key_callback(ord("1"), self.set_rgb_mode)
        vis.register_key_callback(ord("2"), self.set_semantic_mode)
        vis.register_key_callback(ord("3"), self.set_instance_mode)
        
        # Downsampling toggles
        vis.register_key_callback(ord("D"), self.toggle_voxel_downsample)
        vis.register_key_callback(ord("U"), self.toggle_uniform_downsample)
        vis.register_key_callback(ord("]"), self.increase_voxel)
        vis.register_key_callback(ord("["), self.decrease_voxel)
        vis.register_key_callback(ord("="), self.increase_point_size)
        vis.register_key_callback(ord("-"), self.decrease_point_size)
        
        # Tools
        vis.register_key_callback(ord("C"), self.action_center)
        vis.register_key_callback(ord("X"), self.action_crop)
        vis.register_key_callback(ord("R"), self.action_reset)
        vis.register_key_callback(ord("P"), self.action_screenshot)
        
        print("\nControls")
        print("1/2/3 : Color Modes")
        print("D : Toggle Voxel Downsample | U : Toggle Uniform Downsample")
        print("] / [ : Change Voxel Size")
        print("= / - : Change Point Size")
        print("X : Crop in Half | C : Center | R : Reset | P : Screenshot\n")
        
        vis.run()
        vis.destroy_window()

    # Coloring
    def set_rgb_mode(self, vis):
        self.color_mode = "RGB"
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("Mode: RGB")
        
    def set_semantic_mode(self, vis):
        self.color_mode = "SEMANTIC"
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("Mode: SEMANTIC")
        
    def set_instance_mode(self, vis):
        self.color_mode = "INSTANCE"
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("Mode: INSTANCE")

    # Downsampling
    def toggle_voxel_downsample(self, vis):
        self.is_downsampled = not self.is_downsampled
        if self.is_downsampled:
            voxel_downsample(self.pc, self.voxel_size)
            print(f"Voxel Downsample ON (Size: {self.voxel_size:.2f}) - Remaining Points: {len(self.pc['points'])}")
        else:
            self.pc = copy.deepcopy(self.original_pc)
            print(f"Downsample OFF - Points restored to: {len(self.pc['points'])}")
            
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)

    def toggle_uniform_downsample(self, vis):
        self.is_downsampled = not self.is_downsampled
        if self.is_downsampled:
            uniform_downsample(self.pc, self.uniform_k)
            print(f"Uniform Downsample ON (1 in {self.uniform_k}) - Remaining Points: {len(self.pc['points'])}")
        else:
            self.pc = copy.deepcopy(self.original_pc)
            print(f"Downsample OFF - Points restored to: {len(self.pc['points'])}")
            
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)

    def increase_voxel(self, vis):
        self.voxel_size += 0.5  # Increased step size so you see changes faster!
        print(f"Voxel size increased to {self.voxel_size:.2f}")
        if self.is_downsampled:
            self.pc = copy.deepcopy(self.original_pc)
            voxel_downsample(self.pc, self.voxel_size)
            self.update_o3d_geometry()
            vis.update_geometry(self.o3d_pcd)

    def decrease_voxel(self, vis):
        self.voxel_size = max(0.1, self.voxel_size - 0.5)
        print(f"Voxel size decreased to {self.voxel_size:.2f}")
        if self.is_downsampled:
            self.pc = copy.deepcopy(self.original_pc)
            voxel_downsample(self.pc, self.voxel_size)
            self.update_o3d_geometry()
            vis.update_geometry(self.o3d_pcd)
    
    def increase_point_size(self, vis):
        """Increases the visual size of the points on screen."""
        opt = vis.get_render_option()
        opt.point_size += 1.0
        # Force the graphics window to refresh immediately
        vis.update_renderer()
        print(f"Point size increased to: {opt.point_size}")

    def decrease_point_size(self, vis):
        """Decreases the visual size of the points on screen (minimum 1.0)."""
        opt = vis.get_render_option()
        # Use max() to prevent the points from completely disappearing into negative sizes
        opt.point_size = max(1.0, opt.point_size - 1.0)
        vis.update_renderer()
        print(f"Point size decreased to: {opt.point_size}")

    # Tools
    def action_center(self, vis):
        center(self.pc)
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("Point Cloud Centered!")

    def action_crop(self, vis):
        # Dynamically find the exact edges of the current point cloud
        min_b = np.min(self.pc["points"], axis=0)
        max_b = np.max(self.pc["points"], axis=0)
        
        # Change the X boundary to stop exactly halfway through the object
        max_b[0] = (min_b[0] + max_b[0]) / 2.0
        
        crop_axis_aligned(self.pc, min_b, max_b)
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("Point Cloud Cropped")

    def action_reset(self, vis):
        self.pc = copy.deepcopy(self.original_pc)
        self.color_mode = "RGB"
        self.is_downsampled = False
        self.update_o3d_geometry()
        vis.update_geometry(self.o3d_pcd)
        print("View Reset to Original")
        
    def action_screenshot(self, vis):
        """Captures the current view and saves it into a dedicated folder."""
        folder_name = "screenshots"
        
        # Check if the folder exists. If it doesn't, create it instantly!
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created new directory: {folder_name}/")
            
        # Generate the unique filename
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder_name, filename)
        
        # Tell Open3D to take the picture and save it to the exact path
        vis.capture_screen_image(filepath)
        
        print(f"Screenshot successfully saved to: {filepath}")

def get_raw_colors(pc: dict):
    """
    Retrieves the original RGB colors and ensures they are normalized between 0.0 and 1.0 for Open3D.
    """
    colors = pc["colors"]
    
    # If any color value greater than 1.0, the data is in 0-255 format and so divide everything by 255
    if colors.max() > 1.0:
        return colors / 255.0
        
    # If the maximum is already 1.0 or less, they are already perfect
    return colors
    
def color_by_semantic_label(labels):
    """Assigns a unique RGB color between 0.0 and 1.0 per semantic class."""
    new_colors = np.zeros((labels.shape[0], 3))

    color_map = {
        0: [0.5, 0.5, 0.5],   # 0 -> Gray
        7: [1.0, 0.0, 0.0],   # 7 -> Red
        10: [0.0, 1.0, 0.0],  # 10 -> Green
        12: [0.0, 0.0, 1.0],  # 12 -> Blue
        15: [1.0, 1.0, 0.0],  # 15 -> Yellow
        17: [1.0, 0.0, 1.0],  # 17 -> Pink
        18: [0.0, 1.0, 1.0],  # 18 -> Cyan
    }
    
    unique_classes = np.unique(labels)
    
    for class_id in unique_classes:
        mask = (labels == class_id).flatten()
        
        # Convert NumPy ID to a standard Python integer to guarantee dictionary matching
        clean_id = int(class_id)
        
        if clean_id in color_map:
            new_colors[mask] = color_map[clean_id]
        else:
            # Fallback for IDs 20 through 31
            np.random.seed(clean_id * 100)
            new_colors[mask] = np.random.rand(3)
            
    return new_colors

def color_by_instance_id(instance_ids):
    """
    Assigns a random but consistent RGB color between 0.0 and 1.0 to each unique object instance.
    """
    # Create a blank black canvas
    new_colors = np.zeros((instance_ids.shape[0], 3))
    
    # Find out exactly how many unique objects exist in the data
    unique_ids = np.unique(instance_ids)
    
    # Loop through every unique object ID
    for inst_id in unique_ids:
        
        # Lock the random generator to this specific ID
        np.random.seed(int(inst_id))
        
        # Generate 3 random numbers (Red, Green, Blue) between 0.0 and 1.0
        random_color = np.random.rand(3)
        
        # Create a True/False mask for points belonging to this exact object
        mask = (instance_ids == inst_id).flatten()
        
        # Paint the canvas for the True rows
        new_colors[mask] = random_color
        
    return new_colors

def show_only_class(pc, labels, class_id):
    """
    Filters the point cloud in-place, keeping only the points that belong to the specified semantic class.
    """
    # Create a True/False mask for the target class
    mask = (labels == class_id).flatten()
    
    # Loop through every array in our dictionary
    for key in pc.keys():
        
        # Apply the mask to instantly delete the False rows
        pc[key] = pc[key][mask]