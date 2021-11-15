from PIL import Image
import cv2 as cv
import os
import numpy as np




directory = 'masks_no_errosion_abstract/screw_random_dataset_17_2.png'
img = Image.open(directory)
width, height = img.size
new_image = img.resize((width*5, height*5))
new_image.save('blablabla.png')
#Image._show(img)

mask_to_morph = cv.imread('blablabla.png')

        # Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        # img_erosion = cv.erode(img, kernel, iterations=1)
img_erosion = cv.erode(mask_to_morph, kernel, iterations=2)

cv.imwrite('zzblalba.png',
            img_erosion)