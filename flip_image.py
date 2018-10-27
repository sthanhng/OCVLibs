"""
Demo of the flip image
"""

import cv2

from utils.helpers import flip

# Read the image input
image = cv2.imread('./images/dog.jpg')

# Flip the image
flipped_img = flip(image, random_flip=True)

# Display the original and rotated image
wind_name_orig = cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
cv2.imshow('Original image', image)

wind_name_flip = cv2.namedWindow('Flipped image', cv2.WINDOW_NORMAL)
cv2.imshow('Flipped image', flipped_img)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('outputs/flipped_image.jpg', flipped_img)
cv2.destroyAllWindows()
