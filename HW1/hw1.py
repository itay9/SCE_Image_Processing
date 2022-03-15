from cv2 import cv2 as cv
import os
import sys
import imghdr

# getting the source folder from the argument in the command line
folder=os.listdir(sys.argv[1])
# valid picture formats
valid_formats =['rgb', 'gif', 'pbm', 'pgm','ppm','tiff','rast','xbm','jpeg','jpg','bmp','png','webp','exr',]

for image in folder:
    try:
        # check if file in folder is an image
        imghdr.what(sys.argv[1]+'\\'+image) in valid_formats
        # read the image
        img = cv.imread(sys.argv[1]+'\\'+image)
        # transfer it to HSV model
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # lower yellow boundaries in hsv model
        low = (25, 50, 70)
        # upper yellow boundaries in hsv model
        high = (35, 255, 255)
        # creating the mask
        mask = cv.inRange(hsv, low, high)
        # manipulating the image to remove yellow lines
        output = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
        output = cv.bitwise_not(img, mask=cv.bitwise_not(mask))
        output = cv.bitwise_not(output)
        # save the output in given folder
        cv.imwrite(sys.argv[2] + '\\' + image, output)
    except:
        print(image +' is not an image file / supported file')
