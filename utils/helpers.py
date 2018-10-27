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


def build_montages(image_list, image_shape, montage_shape):
    """
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------

    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels

    ---------------------------------------------------------------------------------------------

    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format

    ---------------------------------------------------------------------------------------------

    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)

    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')

    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')

    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                             dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False

    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position

        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(
                    shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                    dtype=np.uint8)
                start_new_img = True

    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage

    return image_montages


def flip(image, random_flip):
    """
    Flip the image
    :param image: The image to flip
    :param random_flip:
    :return: The flipped image
    """

    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)

    return image


def to_rgb(img):
    """
    Convert an image to color space RGB
    :param img: The image need to be converted
    :return: The converted image
    """

    width, height = img.shape
    ret = np.empty((width, height, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img

    return ret


def crop(image, random_crop, image_size):
    """
    Crop the image with or w/o random
    :param image: The image need to be cropped
    :param random_crop: True or False
    :param image_size: The cropped image
    :return:
    """

    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1),
                      np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v),
                (sz1 - sz2 + h):(sz1 + sz2 + h), :]

    return image
