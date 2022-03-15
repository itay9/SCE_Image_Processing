from cv2 import cv2 as cv
import os
import sys
import imghdr
import numpy as np

# getting the source folder from the argument in the command line
folder=os.listdir(sys.argv[1])
# valid picture formats
valid_formats =['rgb', 'gif', 'pbm', 'pgm','ppm','tiff','rast','xbm','jpeg','jpg','bmp','png','webp','exr',]

# this function returns the index of the biggest contour
def max_area_idx(contours):
    a = cv.contourArea(contours[0])
    idx = 0
    for x in range(len(contours)):
        if a < cv.contourArea(contours[x]):
            a = cv.contourArea(contours[x])
            idx = x
    return idx


for image in folder:
    try:
        # check if file in folder is an image
        imghdr.what(sys.argv[1]+'\\'+image) in valid_formats

        # read the image
        img = cv.imread(sys.argv[1]+'\\'+image)
        # transfer it to GRAY
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(imgGray,(7,7), 0)
  
        threshold, res = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        imgCopy= img.copy()
        # finding the contours
        (cnts, _) = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # getting the index of the page contour
        i=max_area_idx(cnts)
        
        
        #approximating the contour to 4 points using approxPolyDP
        epsilon=0.01*cv.arcLength(cnts[i],True)
        approx = cv.approxPolyDP(cnts[i], epsilon, True)
        

        
        # getting the height and width
        x, y, w, h = cv.boundingRect(approx)

        cv.drawContours(imgCopy,[approx],0,(255,255,0),8)

        if approx[0][0].sum() > approx[1][0].sum():
            pts1 = np.float32([approx[1][0], approx[2][0], approx[0][0], approx[3][0]])
        else:
            pts1 = np.float32([approx[0][0], approx[1][0], approx[3][0], approx[2][0]])
        
        pts2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        dst = cv.warpPerspective(img, M, (w, h))

        cv.imwrite(sys.argv[2] + '\\' + image, dst)
    except:
        print(image + ' is not an image file / supported file')


