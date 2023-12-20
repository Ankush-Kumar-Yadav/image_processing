import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image from file
image_path = 'path/to/your/image.jpg'  # Replace with the path to your image file
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# X-Sobel edge detection
sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel_x = np.absolute(sobel_x)
sobel_x_uint8 = np.uint8(abs_sobel_x)

# Y-Sobel edge detection
sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)
abs_sobel_y = np.absolute(sobel_y)
sobel_y_uint8 = np.uint8(abs_sobel_y)

# Canny edge detection
canny_edges = cv2.Canny(original_image, 50, 150)

# Plotting the results
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(sobel_x_uint8, cmap='gray')
plt.title('X-Sobel Edge Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobel_y_uint8, cmap='gray')
plt.title('Y-Sobel Edge Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()
