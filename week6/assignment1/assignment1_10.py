import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img_human = cv2.imread("./images/input.jpg")
img_human = cv2.cvtColor(img_human, cv2.COLOR_BGR2RGB)

# Define range for value to filter only skin
lower_hue = np.array([0, 30, 80])
upper_hue = np.array([16, 150, 255])

# Convert image to HSV color space
img_human_hsv = cv2.cvtColor(img_human, cv2.COLOR_RGB2HSV)

# Filter the image
mask_hsv = cv2.inRange(img_human_hsv, lower_hue, upper_hue)

# Masking the image
masked_image = np.copy(img_human)
masked_image[mask_hsv==0] = [0,0,0]

# Convert mask image to gray scale image
masked_image = cv2.cvtColor(masked_image,cv2.COLOR_RGB2GRAY)

# Use color tresholding to create binary image
_, binary = cv2.threshold(masked_image, 40, 215, cv2.THRESH_BINARY)

# Find contours of image
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Remove not skin contours by size
temp_array = []
for idx_contour in range(len(contours)):
    temp_array.append(not (contours[idx_contour].shape[0] < 400))
    if (contours[idx_contour].shape[0] > 400):
      print(contours[idx_contour].shape)

contours_image = np.copy(img_human)
contours = np.array(contours)
temp_array = np.array(temp_array)
contours = contours[temp_array]

# Draw contours on image
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 2)

# Show draw image
plt.imshow(contours_image, cmap="gray")
plt.title("Human Skin") 
plt.axis('off')
plt.show()