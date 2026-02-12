import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

# Load and convert image
image_path = '4shapes.jpg'
img = Image.open(image_path)
img_array = np.array(img)

# Preprocessing
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = img_array.copy()

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    if area < 1000:
        continue
    
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        vertices = len(approx)
        
        # Classify by vertex count
        if vertices == 3:
            shape_name = "Triangle"
            color = (0, 0, 255)
        elif vertices == 4:
            shape_name = "Rectangle"
            color = (0, 255, 0)
        elif vertices == 5:
            shape_name = "Pentagon"
            color = (255, 255, 0)
        elif vertices == 6:
            shape_name = "Hexagon"
            color = (255, 0, 255)
        else:
            shape_name = "Circle"
            color = (255, 0, 0)
        
        cv2.drawContours(output_image, [approx], -1, color, 3)
        cv2.putText(output_image, shape_name, (cx - 50, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Display result
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(output_rgb)
plt.title('Geometrical Shape Detection and Recognition')
plt.axis('off')
plt.show()
