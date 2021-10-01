# Python program to demonstrate erosion and
# dilation of images.
import cv2 as cv
import numpy as np
from PIL import Image

# Reading the input image
img = cv.imread('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_3/Example_1/final_mask.png')
image_pillow = Image.open('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_3/Example_1/final_mask.png')
height = image_pillow.height
width = image_pillow.width


# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
#img_erosion = cv.erode(img, kernel, iterations=1)
img_dilation = cv.dilate(img, kernel, iterations=1)
print(type(img_dilation))
print(img_dilation)
#img_dilation_of_erosion = cv.dilate(img_erosion, kernel, iterations=1)
#img_erosion_of_dilation = cv.erode(img_dilation, kernel, iterations=2)
#img_dilation_of_erosion_of_ilation = cv.dilate(img_erosion, kernel, iterations=2)
#array = np.reshape(img_dilation, (height, width)).astype(np.uint8)

#im = Image.fromarray(array)
#im.save("C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_3/Example_4/dilation_of_final_mask", im)

cv.imwrite('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_3/Example_1/final_mask_dilation.jpg', img_dilation)



#cv.imshow('Input', img)
#cv.imshow('Erosion', img_erosion)
#cv.imshow('Dilation', img_dilation)
#cv.imshow('Dilation_of_Erosion', img_dilation_of_erosion)
#cv.imshow('Erosion_of_Dilation', img_erosion_of_dilation)
#cv.imshow('dilation_of_erosion_of_Dilation', img_dilation_of_erosion_of_ilation)


#cv.waitKey(0)