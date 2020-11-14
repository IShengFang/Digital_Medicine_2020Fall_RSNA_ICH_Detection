# -*- coding: utf-8 -*-
import numpy as np
# https://www.kaggle.com/reppic/gradient-sigmoid-windowing
# https://www.kaggle.com/akensert/rsna-inceptionv3-keras-tf1-14-0/
# https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing
# https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
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
    return window_image(dcm, window_center, window_width)


def brain_window(dcm):
    return window_image(dcm, 40, 80)


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    bone_img = window_image(dcm, 600, 2600)
    return np.array([brain_img, subdural_img, bone_img])


def all_channel_window(dcm):
    grey_img = brain_window(dcm) * 3.0
    all_channel_img = np.zeros((3, grey_img.shape[0], grey_img.shape[1]))
    all_channel_img[2, :, :] = np.clip(grey_img, 0.0, 1.0)
    all_channel_img[0, :, :] = np.clip(grey_img-1.0, 0.0, 1.0)
    all_channel_img[1, :, :] = np.clip(grey_img-2.0, 0.0, 1.0)
    return all_channel_img


def rainbow_window(dcm):
    grey_img = brain_window(dcm)
    rainbow_img = np.zeros((3, grey_img.shape[0], grey_img.shape[1]))
    rainbow_img[0, :, :] = np.clip(4*grey_img-2, 0, 1.0) * (grey_img>0) * (grey_img<=1.0)
    rainbow_img[1, :, :] = np.clip(4*grey_img*(grey_img <=0.75), 0, 1) + np.clip((-4*grey_img+4)*(grey_img>0.75), 0, 1)
    rainbow_img[2, :, :] = np.clip(-4*grey_img+2, 0, 1.0) * (grey_img>0) * (grey_img<=1.0)
    return rainbow_img
