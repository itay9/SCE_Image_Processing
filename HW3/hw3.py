import cv2
import numpy as np
import sys

def img_resize(img, scale_percent):
    """

    :param img: image to resize
    :param scale_percent:
    :return: scaled image
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    newimg = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return newimg


def resize_right_img(left, right):
    """
    scaling the right image to the left image

    :param left: left img src
    :param right: right img src
    :return: scaled img
    """
    leftHeight = left.shape[0]
    rightHeight = right.shape[0]
    scale_percent = leftHeight / rightHeight
    if scale_percent == 1:
        return right
    new_right = img_resize(right, scale_percent)
    return new_right

img_left_color = cv2.imread(sys.argv[1])
img_right_color = cv2.imread(sys.argv[2])

img_left = cv2.cvtColor(img_left_color,cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(img_right_color,cv2.COLOR_BGR2GRAY)
img_right = resize_right_img(img_left, img_right)

orb = cv2.ORB_create()
keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)
keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)

matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = matcher.match(descriptors_left, descriptors_right)
matches = sorted(matches, key=lambda x: x.distance)  # sort
matches = matches[:int(len(matches) * 0.2)]  # taking 20%


keypoints_left_list = []
keypoints_right_list = []
src,dst = None,None

for match in matches:
    (rowLeft, colLeft) = keypoints_left[match.queryIdx].pt
    (rowRight, colRight) = keypoints_right[match.trainIdx].pt
    keypoints_left_list.append((rowLeft, colLeft))
    keypoints_right_list.append((rowRight, colRight))
    src = np.array(list(keypoints_left_list))
    dst = np.array(list(keypoints_right_list))

H,_ = cv2.findHomography(dst, src, cv2.RANSAC)

res = cv2.warpPerspective(img_right_color, H, (img_right.shape[1]+img_left.shape[1], img_left.shape[0]))
res[:img_left.shape[0],:img_left.shape[1]]=img_left_color

cv2.imwrite(sys.argv[3] + '\\' + 'PanoramaImage.jpg', res)
