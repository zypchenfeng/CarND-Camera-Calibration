import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image and grayscale it
image = mpimg.imread('signs_vehicles_xygrad.png')


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #because this uses mpimg to read image
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient=='x'*1) + int(orient=='y'*0),int(orient=='x'*0) + int(orient=='y'*1))
    # 3) Take the absolute value of the derivative or gradient
    sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255*sobel/np.max(sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    binary_output = np.zeros_like(sobel)
    # is > thresh_min and < thresh_max
    binary_output[(sobel>=thresh_min)&(sobel<=thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img)  # Remove this line

    #     # Convert to grayscale (quiz solution code)
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # Apply x or y gradient with the OpenCV Sobel() function
    #     # and take the absolute value
    #     if orient == 'x':
    #         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    #     if orient == 'y':
    #         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    #     # Rescale back to 8 bit integer
    #     scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    #     # Create a copy and apply the threshold
    #     binary_output = np.zeros_like(scaled_sobel)
    #     # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    #     binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


# Run the function
grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()