import os

import cv2
import numpy as np


img_path = "/D_data/Front/data/VFHQ_datasets_extracted/VFHQ-Test/extracted_cropped_face_results_25_fps/vBn5dQDJsh8/Clip+vBn5dQDJsh8+P0+C2+F5435-5617/00000001.png"

img = cv2.imread(img_path)


def crop_square_ndarray(img, size=256, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[
        int(h / 2 - min_size / 2) : int(h / 2 + min_size / 2),
        int(w / 2 - min_size / 2) : int(w / 2 + min_size / 2),
    ]
    if h >= size:
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return resized


sq_img = crop_square_ndarray(img)

cv2.imwrite("vfhq_sq_img.png", sq_img)
