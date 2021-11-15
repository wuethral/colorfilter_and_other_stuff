import cv2
import os
import numpy as np
import glob

''' 
# makeing video from images
image_folder = 'images'
video_name = 'image_folder/video_images_hsv_filter.avi'


images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

for i in range(len(images)):
    frame = cv2.imread(os.path.join(image_folder, images[i]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
'''
''' 
video_name = 'image_folder/video_images_hsv_filter.avi'
img_array = []

for filename in glob.glob('filter_frames/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('video_name', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

''' 
video=cv2.VideoCapture('angle(20).avi')
image=cv2.imread('green_screen_stuff/nazghul.jpg')
im = cv2.imread('screw_lr_image_20.png')

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
i = 0
while True:

    i += 1
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
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f=frame-res
    #cv2.imshow('F', f)
    #print(f)
    green_screen=np.where(f==0, image, f)
    #cv2.imshow('Res',res)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    frame_name = 'filter_frames/frame_name_' + str(i) + '.png'
    cv2.imwrite(frame_name, frame)
    mask_name = 'filter_masks/mask_name_' + str(i) + '.png'
    cv2.imwrite(mask_name, mask)
    #cv2.imshow('Green Screen', green_screen)
    k=cv2.waitKey(1000)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

