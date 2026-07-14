import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load an image (Replace 'image.jpg' with your image path)
# Create a dummy image if no file exists for demonstration
try:
    img = cv2.imread('image.jpg')
    if img is None:
        raise FileNotFoundError
except:
    # Generate a synthetic image for testing
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(img, (200, 200), (50), (0, 0, 255), -1)

# 2. Basic Image Processing Operations
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale conversion
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # Blurring
edges = cv2.Canny(blurred_img, 50, 150)  # Edge detection

# 3. Feature Representation (Color Histogram)
# Calculate histogram for the blue channel
hist_blue = cv2.calcHist([img], [0], None, [256], [0, 256])

# 4. Feature Extraction (ORB Keypoint Detection)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_img, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

# 5. Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Grayscale")
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Blue Channel Histogram")
plt.plot(hist_blue, color='blue')
plt.xlim([0, 256])

plt.subplot(2, 3, 5)
plt.title("ORB Keypoints")
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
