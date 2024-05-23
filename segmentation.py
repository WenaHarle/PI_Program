import cv2
import numpy as np

def segment(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for green color (tuned for green apples and guavas)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([100, 255, 255])

    # Create a mask for the green color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Apply the mask to the original frame to get the green areas
    result = cv2.bitwise_and(frame, frame, mask=mask_green)
    return result

# Example usage:
# frame = cv2.imread('path_to_image.jpg')
# result = segment(frame)
# cv2.imshow('Segmented Green', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
