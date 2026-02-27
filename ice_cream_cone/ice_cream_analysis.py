import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load and Preprocess Image
image_path = 'softy.jpg'
img = Image.open(image_path)
img_array = np.array(img)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Ice Cream Cone')
plt.axis('off')

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Thresholding to separate cone from background
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

plt.subplot(2, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Binary Threshold')
plt.axis('off')

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.subplot(2, 3, 4)
plt.imshow(morph, cmap='gray')
plt.title('Morphological Closing')
plt.axis('off')

# Edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

plt.subplot(2, 3, 5)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_image = img_array.copy()

print(f"Found {len(contours)} contours")

# Detect cone shape
for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 3000:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 3)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        aspect_ratio = float(h) / w
        print(f"Ice cream cone detected - Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(output_image, "Ice Cream Cone", (cx-80, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display result
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 6)
plt.imshow(output_rgb)
plt.title('Detected Ice Cream Cone')
plt.axis('off')

plt.tight_layout()
plt.show()
