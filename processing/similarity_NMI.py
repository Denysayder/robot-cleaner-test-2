import skimage
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from scipy.stats import entropy
import numpy as np

def compare_images(frame1, frame2):
    if frame1 is None or frame2 is None:
        return 0
    size = (300, 300)
    image1 = cv.resize(frame1, size)
    image2 = cv.resize(frame2, size)
    if (np.max(image1) - np.min(image1)) == 0:
        image1_normalized = image1
    else:
        image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))

    if (np.max(image2) - np.min(image2)) == 0:
        image2_normalized = image2
    else:
        image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    image1_bw = np.where(image1_normalized >= 0.5, 1, 0)
    image2_bw = np.where(image2_normalized >= 0.5, 1, 0)
    similarity = ssim(image1_bw, image2_bw, data_range=1.0)
    se = np.mean((image1_bw - image2_bw) ** 2)
    mse = se**(1 / 2)
    histogram1 = np.histogram(image1_normalized, bins=256)[0]
    histogram2 = np.histogram(image2_normalized, bins=256)[0]
    joint_histogram = np.histogram2d(image1_normalized.flatten(),
                                     image2_normalized.flatten(), bins=256)[0]
    nmi = ((entropy(histogram1) + entropy(histogram2)) / entropy(joint_histogram.flatten()))
    combined_score = (nmi * similarity * (1 - mse))**(1 / 3)
    if np.isnan(combined_score):
        combined_score = 0
    return combined_score