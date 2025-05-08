import skimage
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from scipy.stats import entropy
# from skimage.io import imread
import numpy as np
# import matplotlib.pyplot as plt

def compare_images(frame1, frame2):
    if frame1 is None or frame2 is None:
        return 0
    size = (300, 300)
    image1 = cv.resize(frame1, size)
    image2 = cv.resize(frame2, size)
    # Normalize the pixel values between 0 and 1
    if (np.max(image1) - np.min(image1)) == 0:
        image1_normalized = image1
    else:
        image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))

    if (np.max(image2) - np.min(image2)) == 0:
        image2_normalized = image2
    else:
        image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    # Convert image1 to black and white
    image1_bw = np.where(image1_normalized >= 0.5, 1, 0)
    # Convert image2 to black and white
    image2_bw = np.where(image2_normalized >= 0.5, 1, 0)
    # Calculate the structural similarity index (SSIM)
    similarity = ssim(image1_bw, image2_bw, data_range=1.0)
    # Calculate the Mean square error (MSE)
    se = np.mean((image1_bw - image2_bw) ** 2)
    mse = se**(1 / 2)
    # Calculate the histograms
    histogram1 = np.histogram(image1_normalized, bins=256)[0]
    histogram2 = np.histogram(image2_normalized, bins=256)[0]
    joint_histogram = np.histogram2d(image1_normalized.flatten(),
                                     image2_normalized.flatten(), bins=256)[0]
    # Calculate the normalized mutual information (NMI)
    nmi = ((entropy(histogram1) + entropy(histogram2)) / entropy(joint_histogram.flatten()))
    # Combine the normalized metrics
    combined_score = (nmi * similarity * (1 - mse))**(1 / 3)
    if np.isnan(combined_score):
        combined_score = 0
    return combined_score

# display image
# # Paths to the two black and white images
# image1_path = 'image/fourier_result2.jpg'
# image2_path = 'image/fourier_result3.jpg'
# # Compare the images and get the similarity score, mask, and images

# similarity_score, similarity_mask, image1_bw, image2_bw, image1_normalized, image2_normalized = compare_images(image1_path, image2_path)

# # Create a figure to display the images
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# # Display image1
# axes[0, 0].imshow(image1_normalized, cmap='gray', vmin=0, vmax=1)
# axes[0, 0].axis('off')
# axes[0, 0].set_title('Image 1 (Normalized)')
# # Display image2
# axes[0, 1].imshow(image2_normalized, cmap='gray', vmin=0, vmax=1)
# axes[0, 1].axis('off')
# axes[0, 1].set_title('Image 2 (Normalized)')
# # Display the similarity mask
# axes[1, 0].imshow(similarity_mask, cmap='gray')
# axes[1, 0].axis('off')
# axes[1, 0].set_title('Similar Pixels (White) and Different Pixels (Black)')
# # Add a text box to show the similarity score
# axes[1, 1].text(0.5, 0.5, f'Similarity: {similarity_score:.2f}', ha='center', fontsize=12)
# axes[1, 1].axis('off')
# # Adjust the spacing between subplots
# plt.subplots_adjust(wspace=0.2, hspace=0.4)
# # Show the figure
# plt.show()
