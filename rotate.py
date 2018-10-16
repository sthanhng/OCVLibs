# Demo of the rotate function

import cv2

from utils.helpers import rotate


# read the image input
image = cv2.imread('./images/deadpool.jpg')

# perform the rotation
rotated = rotate(image, angle=60)

# display the rotated image
wind_name = cv2.namedWindow('rotated image', cv2.WINDOW_NORMAL)
cv2.imshow('rotated image', rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()