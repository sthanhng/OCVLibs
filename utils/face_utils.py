# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# A collection of OpenCV libraries
#
# Description : face_utils.py
#
# *******************************************************************


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    #  to the format (x, y, w, h) as we would normally do
    #  with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def bb_to_rect(x, y, w, h):
    # take a bounding box predicted by face detector and convert it
    # to the format (left, top, right, bottom) as we would normally do
    # with dlib
    left = x
    top = y
    right = w + x
    bottom = h + y

    return (left, top, right, bottom)
