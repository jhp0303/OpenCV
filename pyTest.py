from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import pytesseract
import matplotlib.pyplot as plt
plt.style.use('dark_background')



#------------------------------------------------------------------------------------------------------#

# Changing the contrast and brightness of an image

# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()
image = cv.imread("C:/Users/Wolf/Desktop/ALPR/Image/bmw_gray.jpg", cv.IMREAD_UNCHANGED)

if image is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
new_image = np.zeros(image.shape, image.dtype)
alpha = 3.0 # Simple contrast control
beta = 0    # Simple brightness control
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
cv.imshow('Original Image', image)
cv.imshow('New Image', new_image)
cv.imwrite('Image/bmw_maxContrast.jpg', new_image) # Save the New Image
# Wait until user press some key
cv.waitKey()

#------------------------------------------------------------------------------------------------------#

# # Show Video

# capture = cv2.VideoCapture("C:/Users/Wolf/Desktop/ALPR/Video/sampleVideo1.mp4")

# while True:
#     if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
#         capture.open("Video/sampleVideo1.mp4")

#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)

#     if cv2.waitKey(33) > 0: break

# capture.release()
# cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------#

# # Show Image

# image = cv2.imread("C:/Users/Wolf/Desktop/ALPR/Image/overstack.jpg", cv2.IMREAD_UNCHANGED)
# cv2.imshow("Stack", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
