import cv2
import os
import numpy as np

''' 
# makeing video from images
image_folder = 'D:/masks/hands/original_images_renamed'
video_name = 'D:/masks/hands/hands.avi'


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
'''

video=cv2.VideoCapture('D:/masks/hands/hands.avi')
image=cv2.imread('green_screen_stuff/nazghul.jpg')

def nothing():
    pass

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 300, 300)

cv2.createTrackbar('L-H', 'Trackbars', 0,179, nothing)
cv2.createTrackbar('L-S', 'Trackbars', 0,255, nothing)
cv2.createTrackbar('L-V', 'Trackbars', 0,255, nothing)
cv2.createTrackbar('U-H', 'Trackbars', 179,179, nothing)
cv2.createTrackbar('U-S', 'Trackbars', 255,255, nothing)
cv2.createTrackbar('U-V', 'Trackbars', 255,255, nothing)

while True:


    ret, frame = video.read()
    frame = cv2.resize(frame, (640,480))
    image=cv2.resize(image,(640,480))
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos('L-H', 'Trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbars')
    u_h = cv2.getTrackbarPos('U-H', 'Trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbars')

    #l_green=np.array([45, 87, 95])
    #u_green = np.array([90, 255, 255])

    l_green = np.array([l_h, l_s, l_v])
    u_green = np.array([u_h, u_s, u_v])

    mask=cv2.inRange(hsv, l_green, u_green)
    #print(mask)
    #print(mask.shape)
    #print(frame.shape)
    #res = cv2.bitwise_and(frame, frame, mask=mask)
    #f=frame-res
    #cv2.imshow('F', f)
    #print(f)
    #green_screen=np.where(f==0, image, f)
    #cv2.imshow('Res',res)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    #cv2.imshow('Green Screen', green_screen)
    k=cv2.waitKey(1000)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

