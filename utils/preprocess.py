# -*- coding: utf-8 -*-
import pydicom
import numpy as np
# https://www.kaggle.com/reppic/gradient-sigmoid-windowing
# https://www.kaggle.com/akensert/rsna-inceptionv3-keras-tf1-14-0/
# https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
# https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing


def window_image(dcm, window_center, window_width):
    img = dcm.pixel_array*dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img = np.clip(img, img_min, img_max)
    img = (img-img_min) / (img_max-img_min)
    return img


def meta_window(dcm):
    try:
        window_center = float(dcm.WindowCenter)
        window_width = float(dcm.WindowWidth)
    except:
        window_center = float(dcm.WindowCenter[0])
        window_width = float(dcm.WindowWidth[0])
    window_center = int(window_center)
    window_width = int(window_width)
    img = window_image(dcm, window_center, window_width)
    return np.array([img, img, img]).transpose(1, 2, 0)


def brain_window(dcm):
    window_center = 40
    window_width = 80
    img = window_image(dcm, window_center, window_width)
    return np.array([img, img, img]).transpose(1, 2, 0)


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    bone_img = window_image(dcm, 600, 2000)    
    return np.array([brain_img, subdural_img, bone_img]).transpose(1, 2, 0)
