# Demo of the resize image function

import cv2

from utils.helpers import resize


# read the image input
image = cv2.imread('./images/deadpool.jpg')

# perform the rotation
resized = resize(image, resized_width=500)

# display the rotated image
wind_name = cv2.namedWindow('resized image', cv2.WINDOW_NORMAL)
cv2.imshow('resized image', resized)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('resized_image.jpg', resized)

cv2.destroyAllWindows()