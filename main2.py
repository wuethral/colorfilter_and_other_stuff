import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im_green = cv.imread('C:/Users/wuethral/Desktop/cut_darkets.png')
hsv = cv.cvtColor(im_green, cv.COLOR_BGR2HSV)
cv.namedWindow("hsv")
cv.imshow('hsv', hsv)

# red
min_red = im_green[..., 0].min()
max_red = im_green[..., 0].max()
# green
min_green = im_green[..., 1].min()
max_green = im_green[..., 1].max()
# blue
min_blue = im_green[..., 2].min()
max_blue = im_green[..., 2].max()
print('RBG:')
print('min_red:', min_red)
print('max_red:', max_red)
print('min_green:', min_green)
print('max_green:', max_green)
print('min_blue:', min_blue)
print('max_blue:', max_blue)
print('')

# hue
min_hue = hsv[..., 0].min()
max_hue = hsv[..., 0].max()
# saturation
min_saturation = hsv[..., 1].min()
max_saturation = hsv[..., 1].max()
# value
min_value = hsv[..., 2].min()
max_value = hsv[..., 2].max()
print('HSV:')
print('min_hue:', min_hue)
print('max_hue:', max_hue)
print('min_saturation:', min_saturation)
print('max_saturation:', max_saturation)
print('min_value:', min_value)
print('max_value:', max_value)
'''
if cv.waitKey(0):
    cv.destroyAllWindows()


plt.style.use('seaborn-whitegrid')
fig = plt.figure()
plt.xlabel('Min_Value')
plt.ylabel('Max_Value')
plt.title('HSV_plot')
#ax = fig.gca()
plt.xlim(0,255)
plt.ylim(0,255)

plt.plot(42,132,'ro',color='green')
plt.text(42,132,'R',color='red')
plt.plot(95,186,'ro',color='green')
plt.text(95,186,'G',color='green')
plt.plot(0,85,'ro',color='green')
plt.text(0,85,'B',color='blue')

plt.plot(206,255,'ro',color='blue')
plt.text(206,255,'R',color='red')
plt.plot(120,192,'ro',color='blue')
plt.text(120,192,'G',color='green')
plt.plot(0,72,'ro',color='blue')
plt.text(0,72,'B',color='blue')

plt.plot(159,255,'ro',color='silver')
plt.text(159,255,'R',color='red')
plt.plot(129,255,'ro',color='silver')
plt.text(129,255,'G',color='green')
plt.plot(77,205,'ro',color='silver')
plt.text(77,205,'B',color='blue')

plt.plot(51,171,'ro',color='black')
plt.text(51,171,'R',color='red')
plt.plot(57,161,'ro',color='black')
plt.text(57,161,'G',color='green')
plt.plot(17,138,'ro',color='black')
plt.text(17,138,'B',color='blue')

plt.savefig('rgb_plot.png')


plt.plot(71,79,'ro',color='green')
plt.text(71,79,'H',color='red')
plt.plot(136,255,'ro',color='green')
plt.text(136,255,'S',color='green')
plt.plot(95,186,'ro',color='green')
plt.text(95,186,'V',color='blue')

plt.plot(100,106,'ro',color='blue')
plt.text(100,106,'H',color='red')
plt.plot(173,255,'ro',color='blue')
plt.text(173,255,'S',color='green')
plt.plot(206,255,'ro',color='blue')
plt.text(206,255,'V',color='blue')

plt.plot(0,106,'ro',color='silver')
plt.text(0,106,'H',color='red')
plt.plot(0,151,'ro',color='silver')
plt.text(0,151,'s',color='green')
plt.plot(159,255,'ro',color='silver')
plt.text(159,255,'V',color='blue')

plt.plot(80,102,'ro',color='black')
plt.text(80,102,'H',color='red')
plt.plot(45,180,'ro',color='black')
plt.text(45,180,'S',color='green')
plt.plot(57,171,'ro',color='black')
plt.text(57,171,'V',color='blue')

plt.savefig('hsv_plot.png')

plt.grid()
plt.show()

'''