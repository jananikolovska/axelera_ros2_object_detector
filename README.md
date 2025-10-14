# 🤖 Axelera AI x ROS2 Integration

This ROS2 workspace contains packages for integrating Axelera AI inference capabilities with ROS2, specifically designed for YOLO object detection demo using either camera input or streamed video.

## 🎯 Overview

This workspace demonstrates the integration of Axelera AI's hardware acceleration platform with ROS2 for real-time computer vision applications. The workspace is structured to provide both C++ and Python implementations for AI inference nodes, along with supporting helper utilities.

## 📦 Packages

### 1. `ax_inference_node_cpp`
- **Description**: C++ implementation of inference node
- **Type**: C++ package (ament_cmake)
- **Purpose**: Provides high-performance C++ node for running YOLO object detection using Axelera AI hardware
- **Dependencies**: rclcpp, sensor_msgs, cv_bridge, axruntime
- **Launch Files**: `inference_launch.py` - Main launch configuration for the inference demo

### 2. `ax_inference_node_python`
- **Description**: Python implementation of inference node
- **Type**: Python package (ament_python)
- **Purpose**: Provides Python-based node for running YOLO object detection using Axelera AI hardware
- **Dependencies**: rclpy, sensor_msgs, cv_bridge
- **Launch Files**: `inference.launch.py` - Python inference launch configuration

### 3. `helper_nodes`
- **Description**: Supporting utility nodes for input and output processing
- **Type**: Python package (ament_python)
- **Purpose**: Provides auxiliary functionality including:
  - **Input Nodes**: Live camera capture or video streaming to simulate live camera feed
  - **Display Node**: Output processing with options for visual display or text printing
- **Dependencies**: rclpy, sensor_msgs, cv_bridge, opencv-python

## 🏗️ Workspace Structure

```
voyager-sdk/
└── ros2_ws/
    ├── README.md           # This file
    ├── src/                # Source packages directory
    │   ├── ax_inference_node_cpp/     # C++ inference implementation
    │   ├── ax_inference_node_python/  # Python inference implementation
    │   └── helper_nodes/              # Input/output utility nodes
    └──
```

## 🚀 Getting Started

### 📃 Prerequisites
- ROS2 (Humble)
- **Voyager SDK**: Install a virtual environment with the newest version of Voyager SDK from the [Axelera repository](https://github.com/axelera-ai-hub/voyager-sdk)
- Axelera AI SDK and drivers
- OpenCV with Python bindings
- (optional) Camera hardware for live demo

## 🎮 Running the Demo

The inference system consists of modular ROS2 nodes that work together. Each node runs in a separate terminal (this is how ROS2 works - nodes communicate via topics across different processes).

### 🔧 Terminal Setup

**Every time you open a new terminal**, you must first navigate to the ROS2 workspace inside the voyager-sdk folder and source the workspace setup:
```bash
source workspace_setup.sh
```

This script sets up all necessary environment variables and dependencies. You only need to run this **once per terminal session** when you first open it.

For any code changes afterwards, you can use the standard ROS2 workflow:
```bash
colcon build
source install/setup.bash
```

### ⚙️ Launch Files Configuration

All hardcoded values and parameters are configured in the launch files:
- **Model parameters**: model name, AIPU cores, confidence thresholds
- **Topic names**: input/output topic routing
- **Processing parameters**: mean/stddev values, NMS thresholds

### 🕵️‍♀️ Inference Pipeline

**Terminal 1 - Input Node (Choose one):**

For **live camera** input:
```bash
ros2 run helper_nodes camera_node
```

For **video file** streaming (simulates live camera):
```bash
ros2 run helper_nodes video_stream_node
```

**Terminal 2 - Inference Node (Choose one):**

For **C++** infernece node:
```bash
ros2 launch ax_inference_node_cpp inference_launch.py
```
For **Python** inference node:

```bash
ros2 launch ax_inference_node_python inference.launch.py
```

**Terminal 3 - Display Node (Choose one):**

For **visual display** with saved frames:
```bash
ros2 run helper_nodes display_node
```

To see the **text output** topic only:
```bash
ros2 topic echo /detections_topic
```

### 🔧 Package Modularity

The inference nodes (both C++ and Python) are designed as **separate packages** for easy integration:
- **Standalone packages**: Each inference implementation can be taken independently
- **Easy integration**: Simply include the package in your ROS2 workspace
- **Configurable via launch**: All parameters controlled through launch files
- **Topic-based communication**: Standard ROS2 message interfaces

## 📊 Output Formats

The system provides two types of output:

### 1. Text Output (String Messages)
Published on `/detections_topic` with bounding box coordinates:
```
Detection: person (85%) at (245.3, 123.7, 67.2, 156.8)
Detection: car (92%) at (445.1, 267.3, 123.4, 89.6)
Detection: bicycle (78%) at (156.7, 203.2, 45.8, 78.3)
```

Format: `Detection: [class_name] ([confidence]%) at ([x], [y], [width], [height])`

### 2. Visual Output (Annotated Frames)
Published on `/camera_frame_annotated` as image messages:
- **Bounding boxes**: Drawn around detected objects
- **Labels**: Class names with confidence percentages
- **Color coding**: Green for high confidence (≥50%), red for lower confidence
- **Center points**: Red dots marking object centers
- **Model info**: Model name displayed on frame

### Topic Structure

- **Input**: `/camera_frame` (sensor_msgs/Image)
- **Detection Output**: `/detections_topic` (std_msgs/String)
- **Visual Output**: `/camera_frame_annotated` (sensor_msgs/Image)

