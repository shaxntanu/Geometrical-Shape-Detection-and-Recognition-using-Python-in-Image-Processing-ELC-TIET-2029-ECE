import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load and Preprocess Image
image_path = 'watch.jpeg'
img = Image.open(image_path)
img_array = np.array(img)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Smartwatch Image')
plt.axis('off')

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles (watch face and display)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                          param1=100, param2=30, minRadius=50, maxRadius=300)

output_image = img_array.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Detected {len(circles[0])} circular features")
    
    for i, circle in enumerate(circles[0, :]):
        center = (circle[0], circle[1])
        radius = circle[2]
        
        # Draw circle outline
        cv2.circle(output_image, center, radius, (0, 255, 0), 3)
        # Draw center point
        cv2.circle(output_image, center, 2, (255, 0, 0), 3)
        
        print(f"Circle {i+1}: Center=({circle[0]}, {circle[1]}), Radius={radius}")

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

plt.subplot(2, 2, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Display result with detected circles
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 4)
plt.imshow(output_rgb)
plt.title('Detected Circular Features')
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional analysis: Find contours for watch components
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nTotal contours detected: {len(contours)}")
