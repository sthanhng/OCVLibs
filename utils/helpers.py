# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# A collection of OpenCV libraries
#
# Description : helpers.py
#
# *******************************************************************


import numpy as np
import cv2
import os


valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def rotate(img, angle, center=None, scale=1.0):
    # get the dimensions of the image
    (height, width) = img.shape[:2]

    # if the center is None, initialize it as the center of the image
    if center == None:
        center = (width / 2, height / 2)

    # the rotation
    rota_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, rota_matrix, (width, height))

    # return the rotated image
    return rotated_img


def resize(image, resized_width=None, resized_height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the image size
    (height, width) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if resized_width is None and resized_height is None:
        return image

    if resized_width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        ratio = resized_height / float(height)
        dim = (int(width * ratio), resized_height)
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        ratio = resized_width / float(width)
        dim = (resized_width, int(height * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def list_files(base_path, valid_exts=valid_exts, contains=None):
    # Loop over the directory
    for (root_dir, dir_names, file_names) in os.walk(base_path):
        # Loop over the file names in the current directory
        for fn in file_names:
            if contains is not None and fn.find(contains) == -1:
                continue

            # Determine the file extension of the current file
            ext = fn[fn.rfind('.'):].lower()

            # Check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                image_path = os.path.join(root_dir, fn).replace(' ', '\\ ')
                yield image_path


# List all images in a directory
def list_images(base_path, contains=None):
    return list_files(base_path, valid_exts=valid_exts, contains=contains)
