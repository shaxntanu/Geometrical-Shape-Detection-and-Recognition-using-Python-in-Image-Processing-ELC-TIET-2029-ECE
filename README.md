# Image Analysis Project Collection
## ELC Activity - TIET 2029 ECE

This project contains 6 separate image analysis implementations using OpenCV, NumPy, Matplotlib, and Pillow. Each analysis is organized in its own folder with dedicated Python scripts and Jupyter notebooks.

## Project Structure

```
├── 4shapes/                    # Geometric shape detection
│   ├── 4shapes.jpg
│   ├── shape_detection.py
│   └── shape_detection.ipynb
│
├── cookie/                     # Cookie shape analysis
│   ├── cookie.jpg
│   ├── cookie_analysis.py
│   └── cookie_analysis.ipynb
│
├── ice_cream_cone/            # Ice cream cone detection
│   ├── softy.jpg
│   ├── ice_cream_analysis.py
│   └── ice_cream_analysis.ipynb
│
├── traffic_cone/              # Traffic cone detection
│   ├── traffic.jpg
│   ├── traffic_cone_analysis.py
│   └── traffic_cone_analysis.ipynb
│
├── tree/                      # Christmas tree shape detection
│   ├── tree.png
│   ├── tree_analysis.py
│   └── tree_analysis.ipynb
│
├── watch/                     # Smartwatch display analysis
│   ├── watch.jpeg
│   ├── watch_analysis.py
│   └── watch_analysis.ipynb
│
└── requirements.txt           # Shared dependencies for all projects
```

## Analysis Descriptions

### 1. 4shapes - Geometric Shape Detection
Detects and classifies geometric shapes (triangles, rectangles, pentagons, hexagons, circles) using contour approximation and vertex counting.

**Techniques:** Grayscale conversion, binary thresholding, morphological operations, contour detection, vertex-based classification

### 2. Cookie - Circular Shape Analysis
Analyzes cookie shape using circle detection and texture analysis.

**Techniques:** Hough Circle Transform, Gaussian blur, edge detection, texture analysis

### 3. Ice Cream Cone - Cone Shape Detection
Detects ice cream cone shape and calculates geometric properties.

**Techniques:** Binary thresholding, morphological closing, contour detection, aspect ratio calculation

### 4. Traffic Cone - Color-Based Detection
Detects traffic cone using color segmentation for orange body and white stripes.

**Techniques:** HSV color space conversion, color range masking, contour detection, edge detection

### 5. Tree - Tree Shape Detection
Detects the overall Christmas tree shape using color-based segmentation and contour analysis.

**Techniques:** HSV color segmentation, morphological operations, contour detection, shape approximation

### 6. Watch - Circular Feature Detection
Analyzes smartwatch display by detecting circular features.

**Techniques:** Hough Circle Transform, Gaussian blur, edge detection, contour analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shaxntanu/Geometrical-Shape-Detection-and-Recognition-using-Python-in-Image-Processing-ELC-TIET-2029-ECE.git
cd Geometrical-Shape-Detection-and-Recognition-using-Python-in-Image-Processing-ELC-TIET-2029-ECE
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each folder contains two ways to run the analysis:

### Option 1: Python Script
```bash
cd <folder_name>
python <script_name>.py
```

Example:
```bash
cd 4shapes
python shape_detection.py
```

### Option 2: Jupyter Notebook
```bash
cd <folder_name>
jupyter notebook <notebook_name>.ipynb
```

Example:
```bash
cd cookie
jupyter notebook cookie_analysis.ipynb
```

## Dependencies

All projects use the same dependencies (defined in root requirements.txt):
- opencv-contrib-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- Pillow >= 8.0.0

## Course Information

- **Course**: ELC Activity
- **Institution**: TIET (Thapar Institute of Engineering and Technology)
- **Year**: 2029
- **Department**: ECE (Electronics and Communication Engineering)
- **Date**: February 12, 2026

## License

This project is created for educational purposes as part of the TIET ELC curriculum.