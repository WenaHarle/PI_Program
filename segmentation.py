import cv2
import numpy as np
import os

def segment(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definisikan rentang warna biru muda (73AFEE)
    lower_light_blue_1 = np.array([0, 0, 200])
    upper_light_blue_1 = np.array([300, 255, 255])

    # Buat mask untuk warna biru muda (73AFEE)
    mask_light_blue_1 = cv2.inRange(hsv, lower_light_blue_1, upper_light_blue_1)

    # Definisikan rentang warna biru muda (6294C3)
    lower_light_blue_2 = np.array([100, 100, 100])
    upper_light_blue_2 = np.array([120, 255, 255])

    # Buat mask untuk warna biru muda (6294C3)
    mask_light_blue_2 = cv2.inRange(hsv, lower_light_blue_2, upper_light_blue_2)

    # Gabungkan mask untuk warna biru muda (73AFEE) dan (6294C3)
    final_mask_light_blue = cv2.bitwise_or(mask_light_blue_1, mask_light_blue_2)

    # Inversi mask untuk mendapatkan area non-putih, non-biru muda
    mask_inverse = cv2.bitwise_not(final_mask_light_blue)

    # Gabungkan frame asli dengan mask_inverse untuk mendapatkan area buah
    result = cv2.bitwise_and(frame, frame, mask=mask_inverse)
    return result

