# Point Cloud Processing & Interactive Visualization
## Overview
This project is a custom, lightweight 3D point cloud processing framework built in Python using Open3D and NumPy. It mimics real-world 3D perception pipelines used in autonomous driving, robotics, and simulation by processing `.ply` files containing geometry, colors, semantic labels, and instance IDs.

## Features Implemented
* **Robust Data Loading:** Safely extracts 3D positions, RGB colors, semantic labels, and instance IDs into a centralized dictionary architecture, complete with NaN-validation and shape-consistency checks.
* **Core Processing Engine:** 
  * Linear transformations (Scaling, Translation, Centering).
  * Axis-aligned bounding box (AABB) cropping using boolean masking.
  * Dual-method downsampling: Voxel Grid downsampling and Uniform downsampling.
* **Semantic & Instance Segmentation:** Dynamic color mapping that parses underlying object IDs. It utilizes hardcoded dictionary mapping for known semantic classes while implementing a randomized seed fallback for undiscovered objects.
* **Interactive 3D UI:** A state-based class wrapper around Open3D's `VisualizerWithKeyCallback` that allows live, interactive toggling of features without permanently destroying the original data in memory.

## Folder Structure
```text
assignment_pointcloud/
│
├── data/
│   └── scene.ply             # 3D Point Cloud Data
│
├── src/
│   ├── data_io.py            # Loading and validation functions
│   ├── processing.py         # Mathematical transformations and downsampling
│   ├── visualization.py      # Color mapping and the Interactive UI Class
│   └── main.py               # Main execution pipeline
│
└── README.md
```

## Prerequisites & Installation
This project relies on Python 3.9+, Open3D, and NumPy.
To run the application, activate the dedicated Conda environment on Windows:

## How to run
Trigger the main pipeline from your terminal. The script will automatically calculate absolute paths to locate your scene.ply file safely.

python src/main.py

## Interactive Controls
Once the 3D viewer launches, use the following keyboard shortcuts to interact with the environment:

| Key | Action | Description |
| :--- | :--- | :--- |
| **1** | RGB Mode | Displays the raw, original colors of the point cloud. |
| **2** | Semantic Mode | Colors objects by category (e.g., Floor, Forklift, Box). |
| **3** | Instance Mode | Assigns a consistent, unique random color to every individual object. |
| **D** | Voxel Downsample | Toggles Voxel-based downsampling on/off. |
| **U** | Uniform Downsample | Toggles Uniform downsampling on/off. |
| **] / [** | Scale Voxel | Increases or decreases the voxel size dynamically in real-time. |
| **C** | Center | Moves the entire point cloud centroid to the origin (0, 0, 0). |
| **X** | Crop | Slices the point cloud perfectly in half along the X-axis. |
| **R** | Reset | Restores the original point cloud data from the deep-copied vault. |
