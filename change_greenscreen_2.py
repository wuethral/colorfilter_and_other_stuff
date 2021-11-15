import cv2
import os
import numpy as np
from PIL import Image


#video=cv2.VideoCapture('background_videos/random_dataset.avi')

#video=cv2.VideoCapture('D:/masks/angle(video20)/angle(20).avi')
''' 

def create_new_mask(frame, mask):

    #res_in_method = np.zeros(921600).reshape(480, 640, 3)
    #cv2.imshow('resinmethod', res_in_method)
    res_in_method = frame
    print(mask)
    #cv2.imwrite('z4.png', frame)
    #cv2.imwrite('z5.png', mask)
    for i in range(480):
        for j in range(640):
            if mask[i,j][0] == 0 and mask[i,j][1] == 0 and mask[i,j][2] == 0:
                res_in_method[i,j][0] = 0
                res_in_method[i, j][1] = 0
                res_in_method[i, j][2] = 0
    #\cv2.imwrite('blabla.png', res_in_method)

            #if mask[i,j][0] == 0 and mask[i,j][1] == 0 and mask[i,j][2] == 0:
          #  res_in_method[i,j] = frame[i,j]


    #cv2.imshow('framiframe', frame)


    return res_in_method
'''

''' 

background = cv2.imread('green_screen_stuff/nazghul.jpg')
background = cv2.resize(background,(640,480))
image = cv2.imread('D:/masks/spire1(video23)_2/original_images_renamed/image_1.png')
image = cv2.resize(image,(640,480))
image_clone = image.copy()
mask = cv2.imread('D:/masks/spire1(video23)_2/hsv_dbscan_morph_holes_filled(no_canny)/mask_hsv_dbscan_morph_holels_filled1.png')
mask = cv2.resize(mask,(640,480))

res = create_new_mask(image, mask)
green_screen=np.where(res==0, background, res)
cv2.imwrite('green.png', green_screen)
'''
set_nr = 0
video_shift = 0
list_of_backgrounds = []

for i in range(1):
    video = cv2.VideoCapture('background_videos/pink.mp4')
    for i in range(10):
        ret, background = video.read()

        list_of_backgrounds.append(background)


    #set_nr += 1
    #video_shift += 10
    #img_nr = 1

    #image_names_list = ['angle', 'hand', 'pliers', 'screw', 'screwdriver', 'spire1', 'spire2', 'zz_test_angle', 'zz_test_hand', 'zz_test_pliers',
    #                    'zz_test_screw', 'zz_test_screwdriver', 'zz_test_spire1', 'zz_test_spire2']

    image_name_list = os.listdir('Testing_Data_With_Augmentation/images')
    ''' 
    for i in range(video_shift):
        ret, background = video.read()
    '''
    i = 0
    counter = 1
    for image_name in image_name_list:
        print(image_name)
        #img_nr = 1
        #while True:
        #print(img_nr)
        ''' 
        for i in range(100):
            ret, background = video.read()
        '''
        #ret, background = video.read()
        background = list_of_backgrounds[i]
        background = cv2.resize(background, (1920,1080))
        #image_path = 'Testing_Data_With_Augmentation/images/' + image_name + '_image_' + str(img_nr) + '.png'
        image_path = 'Testing_Data_With_Augmentation/images/' + image_name
        image = cv2.imread(image_path)

        image=cv2.resize(image,(1920,1080))
        hsv=cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        #if image_name == 'pliers':
        #    mask_path = 'PennFudanPed/PedMasks/' + image_name +'_mask_' + str(img_nr) + '_dilated.png'
         #elif image_name == 'zz_test_pliers':
        #    mask_path = 'PennFudanPed/PedMasks/' + image_name + '_mask_' + str(img_nr) + '_dilated.png'
        #else:
        #    mask_path = 'PennFudanPed/PedMasks/' + image_name +'_mask_' + str(img_nr) + '.png'

        mask_path = 'Testing_Data_With_Augmentation/new_masks/' + image_name[:-4] + '_mask.png'
        print(mask_path)
        if counter == 1:
            counter = 10

        else:
            counter += 1
        mask = cv2.imread(mask_path)
        #print(mask.shape)
        mask=cv2.resize(mask,(1920,1080))

            #mask_matrix = np.zeros((480, 640))
            #for i in range(480):
            #    for j in range(640):
            #        if mask[i,j].any == 0:
            #            mask_matrix[i, j] = 0
            #        else:
            #            mask_matrix[i, j] = 255

            #for i in range(1000):
            #    for j in range(1000):
            #        print(mask[i][j])
            #print(frame.shape)
            #print(mask.shape)
            #res = create_new_mask(image, mask)

        green_screen=np.where(mask==0, background, image)

            #cv2.imshow('Mask',mask)
            #cv2.imshow('Res',res)
            #cv2.imshow('Frame', background)
            #cv2.imshow('Green Screen', green_screen)
        green_screen_img_path = 'Testing_Data_With_Augmentation/images_new_backgrounds/' + image_name[:-4] + '_random_dataset.png'
        cv2.imwrite(green_screen_img_path, green_screen)
            #green_screen_img_path_2 = 'masks_no_errosion_bl/' + image_name + '_bl_image_' + str(img_nr) + '.png'
            #cv2.imwrite(green_screen_img_path_2, res)
        k=cv2.waitKey(1000)
        if k==ord('q'):
            break

        i += 1
        if i == 9:
            i = 0
            ''' 
            img_nr += 1
            if img_nr == 7 and image_name.startswith('zz'):
                break

            if img_nr == 31:
                break
            '''
    video.release()
    cv2.destroyAllWindows()

