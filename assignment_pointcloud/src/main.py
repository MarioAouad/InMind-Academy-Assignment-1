# Import the data loader
from data_io import load_pointcloud, validate_pointcloud

# Import the UI viewer
from visualization import PointCloudViewer

def main():
    # Path to point cloud file
    file_path = "data/scene.ply"
    
    print("Loading point cloud...")
    
    # Load the data
    pc_data = load_pointcloud(file_path)
    
    # Validate it to make sure the data is healthy before viewing
    validate_pointcloud(pc_data)
    
    print("\nStarting the 3D Viewer...")
    
    # Initialize the viewer with the loaded dictionary
    viewer = PointCloudViewer(pc_data)
    
    viewer.run()

if __name__ == "__main__":
    main()