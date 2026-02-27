import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load and Preprocess Image
image_path = 'tree.png'
img = Image.open(image_path)
img_array = np.array(img)

# Handle RGBA if present
if img_array.shape[2] == 4:
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Christmas Tree')
plt.axis('off')

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Detect tree shape using color segmentation
hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

# Detect green tree parts
lower_green = np.array([35, 30, 30])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

# Find contours on the green mask
contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = img_array.copy()
tree_detected = False

# Find the largest contour (the tree)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Draw the tree contour
    if area > 5000:  # Filter small noise
        cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 3)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        tree_detected = True
        print(f"Tree shape detected with area: {area:.0f} pixels")

# Detect edges for visualization
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

plt.subplot(2, 3, 3)
plt.imshow(mask_green, cmap='gray')
plt.title('Green Tree Mask')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Approximate the tree shape
if tree_detected:
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    shape_img = np.zeros_like(gray)
    cv2.drawContours(shape_img, [approx], -1, 255, -1)
    plt.subplot(2, 3, 5)
    plt.imshow(shape_img, cmap='gray')
    plt.title('Simplified Tree Shape')
    plt.axis('off')
else:
    plt.subplot(2, 3, 5)
    plt.imshow(gray, cmap='gray')
    plt.title('No Shape Detected')
    plt.axis('off')

# Display result
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 6)
plt.imshow(output_rgb)
plt.title('Tree Shape Detected' if tree_detected else 'No Tree Detected')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\nAnalysis Summary:")
print(f"- Tree shape detected: {'Yes' if tree_detected else 'No'}")
