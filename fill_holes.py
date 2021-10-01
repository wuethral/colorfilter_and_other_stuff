import cv2
import numpy as np
'''
def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

if __name__ == '__main__':
    mask_in = cv2.imread('mask_hsv_canny_dbscan_2.png', 0)
    mask_out = FillHole(mask_in)
    cv2.imwrite('mask_out.png', mask_out)
'''
x = [1,2,3,4,5,6,7,8]
print(x[:-3])