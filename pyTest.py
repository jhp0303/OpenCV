from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import pytesseract
import matplotlib.pyplot as plt
plt.style.use('dark_background')

img = cv.imread("C:/Users/Wolf/Desktop/ALPR/Image/bmw_Thresholding.jpg", cv.IMREAD_UNCHANGED)

img2 = cv.imread("C:/Users/Wolf/Desktop/ALPR/Image/bmw_gray.jpg", cv.IMREAD_UNCHANGED)
blur = cv.GaussianBlur(img2,(3,3),0)
canny=cv.Canny(blur,100,200)

cnts,contours,hierarchy  = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

box1 = []
f_count = 0
select = 0
plate_width = 0

for i in range(len(contours)):
    cnt=contours[i]          
    area = cv.contourArea(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    rect_area=w*h  #area size
    aspect_ratio = float(w)/h # ratio = width/height
        
    if  (aspect_ratio>=0.2)and(aspect_ratio<=1.0)and(rect_area>=100)and(rect_area<=700): 
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        box1.append(cv.boundingRect(cnt))
         
for i in range(len(box1)): ##Buble Sort on python
    for j in range(len(box1)-(i+1)):
        if box1[j][0]>box1[j+1][0]:
                temp=box1[j]
                box1[j]=box1[j+1]
                box1[j+1]=temp
                
#to find number plate measureing length between rectangles
for m in range(len(box1)):
    count=0
    for n in range(m+1,(len(box1)-1)):
        delta_x=abs(box1[n+1][0]-box1[m][0])
        if delta_x > 150:
                break
        delta_y =abs(box1[n+1][1]-box1[m][1])
        if delta_x ==0:
                delta_x=1
        if delta_y ==0:
                delta_y=1           
        gradient =float(delta_y) /float(delta_x)
        if gradient<0.25:
            count=count+1
    #measure number plate size         
    if count > f_count:
        select = m
        f_count = count
        plate_width=delta_x
cv.imwrite('Image/bmw_rectangle.jpg',img)
cv.imshow('bmw_rectangle', img)
cv.waitKey()

#------------------------------------------------------------------------------------------------------#

# Adaptive Thresholding Image

# img = cv.imread("C:/Users/Wolf/Desktop/ALPR/Image/bmw_maxContrast.jpg", cv.IMREAD_UNCHANGED)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# cv.imshow('BINARY_INV', thresh2)
# cv.imwrite('Image/bmw_Thresholding.jpg', thresh2) # Save the New Image
# plt.show()
# cv.waitKey()


#------------------------------------------------------------------------------------------------------#

# Changing the contrast and brightness of an image

# # Read image given by user
# parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
# args = parser.parse_args()
# image = cv.imread("C:/Users/Wolf/Desktop/ALPR/Image/bmw_gray.jpg", cv.IMREAD_UNCHANGED)

# if image is None:
#     print('Could not open or find the image: ', args.input)
#     exit(0)
# new_image = np.zeros(image.shape, image.dtype)
# alpha = 3.0 # Simple contrast control
# beta = 0    # Simple brightness control
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for c in range(image.shape[2]):
#             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
# cv.imshow('Original Image', image)
# cv.imshow('New Image', new_image)
# cv.imwrite('Image/bmw_maxContrast.jpg', new_image) # Save the New Image
# # Wait until user press some key
# cv.waitKey()

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
