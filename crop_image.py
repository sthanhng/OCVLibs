"""
Demo of the crop image
"""

import cv2

from utils.helpers import crop

# Read the image input
image = cv2.imread('./images/dog.jpg')

# Flip the image
cropped_img = crop(image, False, 300)

# Display the original and cropped image
wind_name_orig = cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
cv2.imshow('Original image', image)

wind_name_flip = cv2.namedWindow('Cropped image', cv2.WINDOW_NORMAL)
cv2.imshow('Cropped image', cropped_img)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('outputs/cropped_image.jpg', cropped_img)
cv2.destroyAllWindows()
