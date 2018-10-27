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


import numpy as np


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


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding
    and get a euclidean distance for each comparison face.
    The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order
    as the 'faces' array
    """

    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see
    if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding
    to compare against the list
    :param tolerance: How much distance between faces to consider it a match.
    Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings
    match the face encoding to check
    """

    return list(face_distance(known_face_encodings, face_encoding_to_check)
                <= tolerance)


def shape_to_points(shape, dtype="int"):
    """
    Convert the landmark (x, y)-coordinates to a NumPy array
    :param shape: The list of landmarks (x, y)--coordinates
    :param dtype: Type of the landmarks (x, y)--coordinates
    :return: A Numpy array contains facial landmarks
    """

    # Initialize the list of (x, y)-coordinates
    points = np.zeros((shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        points[i] = (shape.part(i).x, shape.part(i).y)

    # Return the list of (x, y)-coordinates
    return points