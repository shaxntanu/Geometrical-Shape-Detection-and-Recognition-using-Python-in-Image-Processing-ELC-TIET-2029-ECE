import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Load and Preprocess Image
image_path = 'cookie.jpg'
img = Image.open(image_path)
img_array = np.array(img)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Cookie Image')
plt.axis('off')

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Detect circles (cookie shape)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                          param1=50, param2=30, minRadius=50, maxRadius=300)

output_image = img_array.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Detected {len(circles[0])} circular cookie(s)")
    
    for i, circle in enumerate(circles[0, :]):
        center = (circle[0], circle[1])
        radius = circle[2]
        
        # Draw circle outline
        cv2.circle(output_image, center, radius, (0, 255, 0), 4)
        # Draw center point
        cv2.circle(output_image, center, 2, (255, 0, 0), 5)
        
        print(f"Cookie {i+1}: Center=({circle[0]}, {circle[1]}), Radius={radius}")
        
        # Calculate area
        area = np.pi * (radius ** 2)
        print(f"Cookie area: {area:.2f} pixels²")
        
        cv2.putText(output_image, "Cookie", (center[0]-40, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

# Edge detection
edges = cv2.Canny(blurred, 30, 100)

plt.subplot(2, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

plt.subplot(2, 3, 4)
plt.imshow(thresh, cmap='gray')
plt.title('Binary Threshold')
plt.axis('off')

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.subplot(2, 3, 5)
plt.imshow(morph, cmap='gray')
plt.title('Morphological Closing')
plt.axis('off')

# Display result
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.subplot(2, 3, 6)
plt.imshow(output_rgb)
plt.title('Detected Cookie Shape')
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional texture analysis
print("\nTexture Analysis:")
print(f"Mean intensity: {np.mean(gray):.2f}")
print(f"Standard deviation: {np.std(gray):.2f}")
