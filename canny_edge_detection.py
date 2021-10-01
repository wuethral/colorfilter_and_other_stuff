import cv2 as cv

img = cv.imread('C:/Users/wuethral/Desktop/colorfilter_2/final_mask_with_morph_60.png')
scale_percent = 30  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
# Read the original image
# Display original image
cv.imshow('Original', img)
cv.waitKey(0)

# Convert to graycsale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv.imshow('Sobel X', sobelx)
cv.waitKey(0)
cv.imshow('Sobel Y', sobely)
cv.waitKey(0)
cv.imshow('Sobel X Y using Sobel() function', sobelxy)
cv.waitKey(0)

# Canny Edge Detection
#edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
#cv.imwrite("C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_4/mask_canny_edges.png", edges)


# Display Canny Edge Detection Image
cv.imshow('Canny Edge Detection', edges)
cv.waitKey(0)
cv.destroyAllWindows()

#img = cv.imread('C:/Users/wuethral/Desktop/colorfilter_2/14.9.21_try_2/Example_4/pliers.png')
#canny_edge_det(img)