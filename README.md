# DetectIt - Real-time Object Detection PoC App

## Overview

This project is a **Proof-of-Concept (PoC)** desktop application developed to demonstrate real-time object detection using a standard webcam. It uses:

*   **C++20**
*   **Qt 6 Widgets** for the graphical user interface.
*   **OpenCV 4** for webcam access and image handling.
*   **ONNX Runtime** for running the object detection model.
*   **YOLOv8n** (pre-trained on COCO dataset) as the object detection model (`yolov8n.onnx`).

The primary goals of this PoC were to:

1.  Successfully load and run the YOLOv8n ONNX model on cpu within a C++/Qt application.
2.  Capture frames from a webcam using OpenCV.
3.  Display the video feed and overlay the detected bounding boxes and class labels.
4.  Implement basic Frames Per Second (FPS) measurement to gauge performance.

## Purpose: Proof-of-Concept

It is important to note that this application is **intended as a demonstration and technical exploration**.

Areas for potential future development based on this PoC could include:

*   UI/UX improvements (e.g., status bar, model selection, confidence threshold adjustment).
*   Performance optimization (e.g., different ONNX Execution Providers like CUDA, different model sizes).
*   Video file input.
*   Saving detection results.
*   Packaging and distribution.

## Quick Instructions

### Generating the `yolov8n.onnx` Model
1. **Clone the YOLOv8 Repository**:
   ```bash
   git clone https://github.com/ultralytics/yolov8.git
   cd yolov8
   ```
2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Export the Model**:
   ```bash
   python export.py --weights yolov8n.pt --img-size 640 --batch-size 1 --device 0 --include onnx
   ```

### Downloading ONNX Runtime Binaries
1. **Visit the ONNX Runtime Releases Page**: [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
2. **Download the Appropriate Version**: Choose the version for your system architecture (e.g., `onnxruntime-linux-x64-VERSION.zip`).
3. **Extract the Binaries**: Unzip the file and place the `onnxruntime-linux-x64-VERSION` directory in your project root.

### Installing Required Libraries
```bash
sudo apt-get update
sudo apt-get install libopencv-dev libonnx-dev
```

### Creating the `class.names` File
1. **Create a New File**:
   ```bash
   touch data/class.names //class names contains object names like "cup, phone, person, etc."
   ```

## Building and Running

(Assuming prerequisites like Qt6, OpenCV 4, CMake, C++ compiler, and downloaded ONNX Runtime are met)

2.  **Create Build Directory:** `mkdir build && cd build`
3.  **Run CMake:** `cmake ../DetectIt` (pointing CMake to the directory containing the main `CMakeLists.txt`)
4.  **Build:** `make` (or `ninja`)
5.  **Run:** `./DetectIt` (from within the `build` directory)

