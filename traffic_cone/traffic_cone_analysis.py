import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load and Preprocess Image
image_path = 'traffic.jpg'
img = Image.open(image_path)
img_array = np.array(img)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Traffic Cone')
plt.axis('off')

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Color segmentation for orange cone
hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

# Detect orange/red color
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([20, 255, 255])
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

plt.subplot(2, 3, 3)
plt.imshow(mask_orange, cmap='gray')
plt.title('Orange Color Mask')
plt.axis('off')

# Detect white stripes
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
mask_white = cv2.inRange(hsv, lower_white, upper_white)

plt.subplot(2, 3, 4)
plt.imshow(mask_white, cmap='gray')
plt.title('White Stripes Mask')
plt.axis('off')

# Edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

plt.subplot(2, 3, 5)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Find contours
contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_image = img_array.copy()

print(f"Found {len(contours)} contours")

# Detect cone shape
for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 5000:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 3)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        aspect_ratio = float(h) / w
        print(f"Cone detected - Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")
        
        cv2.putText(output_image, "Traffic Cone", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display result
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 6)
plt.imshow(output_rgb)
plt.title('Detected Traffic Cone')
plt.axis('off')

plt.tight_layout()
plt.show()
