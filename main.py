#import the libraries
import cv2 as cv
import numpy as np
from PIL import Image
from canny_edge_detection import canny_edge_det

#read the image
img = cv.imread('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/spaten.png')


scale_percent = 30  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

#convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#set the lower and upper bounds for the green hue
green_blue_lower = np.array([150,0,0])
green_blue_higher = np.array([255,255,255])

green_silver_lower = np.array([0,0,100])
green_silver_higher = np.array([255,255,255])

green_black_lower = np.array([51,0,85])
green_black_higher = np.array([255,161,255])

green_black_lower_hsv = np.array([0,0,0])
green_black_higher_hsv = np.array([79,255,255])


#lower_hsv = np.array([0,0,0])
#higher_hsv = np.array([360,130,360])



#create a mask for green colour using inRange function

mask_green_blue = cv.inRange(img, green_blue_lower, green_blue_higher)
mask_green_silver = cv.inRange(img, green_silver_lower, green_silver_higher)
mask_green_black = cv.inRange(img, green_black_lower, green_black_higher)
mask_green_black_hsv = cv.inRange(hsv, green_black_lower_hsv, green_black_higher_hsv)
cv.imwrite('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_blue.png', mask_green_blue)
cv.imwrite('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_silver.png', mask_green_silver)
cv.imwrite('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_black_hsv.png', mask_green_black_hsv)
cv.imwrite('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/original_img.png', img)


def adding_pixel_values_row(im_matrix, height, column, row_canny_edge_det, row_green_black, width):
    new_row_image_matrix = [0] * width
    #im_matrix = np.zeros((height, width))
    for i in range(width):
        if row_green_black[i] == 0 or row_canny_edge_det[i] == 255:
            new_row_image_matrix[i] = 255
    np_new_row = np.array(new_row_image_matrix)
    return np_new_row

    #im_matrix[column,:] = np_new_row
    #if column == (height-1):
    #    return im_matrix
''' 
matrix = np.zeros((n,2)) # Pre-allocate matrix
for i in range(1,n):
    matrix[i,:] = [3*i, i**2]
print(matrix)
'''

def merging_images(m_canny_edge_det, m_green_black, height, width):
    array_canny_edge_det = np.array(m_canny_edge_det)
    array_green_blue = np.array(m_green_blue)
    array_green_silver = np.array(m_green_silver)
    array_green_black = np.array(m_green_black)
    im_matrix = np.zeros((height, width))

    for i in range(height):
        numpy_new_row = adding_pixel_values_row(im_matrix, height, i,array_canny_edge_det[i], array_green_black[i], width)
        im_matrix[i,:] = numpy_new_row

    return im_matrix

m_green_blue = Image.open('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_blue.png')
m_green_silver = Image.open('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_silver.png')
m_green_black = Image.open('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_green_black_hsv.png')
canny_edge_det(img)
m_canny_edge_det = Image.open('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_canny_edges.png')

height = m_green_blue.height
width = m_green_blue.width

image_as_np_matrix = merging_images(m_canny_edge_det, m_green_black, height, width)
matrix_to_array = np.squeeze(np.asarray(image_as_np_matrix))
array = np.reshape(matrix_to_array, (height, width)).astype(np.uint8)

im = Image.fromarray(array)
im.save("C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/final_mask.png")


#mask_2 = cv.inRange(hsv, lower_hsv, higher_hsv)





#mask2 = cv.inRange(img, lower_rgb, higher_rgb)
#perform bitwise and on the original image arrays using the mask
#res = cv.bitwise_and(img, img, mask=mask)

#create resizable windows for displaying the images
#cv.namedWindow("res", cv.WINDOW_NORMAL)
#cv.namedWindow("hsv", cv.WINDOW_NORMAL)
#cv.namedWindow("mask", cv.WINDOW_NORMAL)
#cv.namedWindow("original")
#cv.namedWindow("mask_green_blue")
#cv.namedWindow("mask_green_silver")
#cv.namedWindow("mask_green_black")
#cv.namedWindow("mask_green_black_hsv")


#cv.namedWindow("mask2")

#cv.imshow('img', img)

#display the images
#cv.imshow("mask", mask)
#cv.imshow("hsv", hsv)
#cv.imshow("res", res)
#cv.imshow("original", img)
#cv.imshow("mask_green_blue", mask_green_blue)
#cv.imshow("mask_green_silver",mask_green_silver)
#cv.imshow("mask_green_black",mask_green_black)
#cv.imshow("mask_green_black_hsv", mask_green_black_hsv)
#cv.imshow("mask2", mask_2)


if cv.waitKey(0):
    cv.destroyAllWindows()