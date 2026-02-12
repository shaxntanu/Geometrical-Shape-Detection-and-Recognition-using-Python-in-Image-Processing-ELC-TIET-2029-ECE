# Geometrical Shape Detection and Recognition using Python
## ELC Activity - TIET 2029 ECE

This project implements a computer vision pipeline to detect and recognize geometrical shapes (Triangles, Rectangles, Pentagons, Hexagons, and Circles) from images using OpenCV, NumPy, Matplotlib, and Pillow.

## Features

- **Image Preprocessing**: Grayscale conversion and binary thresholding
- **Morphological Operations**: Noise reduction and edge refinement using closing operations
- **Contour Detection**: Automatic shape boundary detection
- **Shape Recognition**: Vertex-based classification system
- **Visual Output**: Color-coded shape labeling with matplotlib visualization

## Project Structure

```
├── shape_detection.py          # Python script version
├── shape_detection.ipynb       # Jupyter notebook with step-by-step cells
├── 4shapes.jpg                 # Sample input image
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

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

### Option 1: Run Python Script
```bash
python shape_detection.py
```

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook shape_detection.ipynb
```

The notebook is organized into 4 cells:
- **Cell 1**: Import libraries
- **Cell 2**: Load image, preprocess, and display grayscale
- **Cell 3**: Detect and classify shapes
- **Cell 4**: Display final result with labeled shapes

## How It Works

### 1. Preprocessing
- Load image and convert to grayscale
- Apply inverse binary thresholding (threshold=127)

### 2. Morphological Operations
- Use 3x3 rectangular kernel
- Apply closing operation (2 iterations) to connect edges

### 3. Shape Detection
- Find contours using `cv2.findContours()`
- Approximate contours with `cv2.approxPolyDP()`
- Filter out noise (area < 1000 pixels)

### 4. Shape Recognition
Based on vertex count:
- **3 vertices** → Triangle (Red)
- **4 vertices** → Rectangle (Green)
- **5 vertices** → Pentagon (Cyan)
- **6 vertices** → Hexagon (Magenta)
- **Other** → Circle (Blue)

### 5. Visualization
- Draw colored contours around detected shapes
- Label each shape at its centroid
- Display using matplotlib

## Dependencies

- `opencv-contrib-python` >= 4.5.0
- `numpy` >= 1.19.0
- `matplotlib` >= 3.3.0
- `Pillow` >= 8.0.0

## Sample Output

The program processes the input image and outputs:
- Grayscale conversion visualization
- Detected shapes with color-coded labels
- Console output showing detected shape types and vertex counts

## Troubleshooting

**Issue**: Shapes not detected correctly
- **Solution**: Adjust the threshold value (line with `cv2.threshold()`)
- Try values between 100-150 depending on image brightness

**Issue**: Too many false detections
- **Solution**: Increase the minimum area filter (currently 1000)

**Issue**: Vertex count incorrect
- **Solution**: Adjust epsilon value in `cv2.approxPolyDP()` (currently 0.04)

## Course Information

- **Course**: ELC Activity
- **Institution**: TIET (Thapar Institute of Engineering and Technology)
- **Year**: 2029
- **Department**: ECE (Electronics and Communication Engineering)

## License

This project is created for educational purposes as part of the ELC curriculum.

## Author

Created for ELC Activity 2029 - ECE Department, TIET

**Date**: February 12, 2026
