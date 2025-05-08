import numpy as np
import cv2
def extract_spectrum(image):
load_image = cv2.imread(image)
spectrum = np.abs(np.fft.fftshift(np.fft.fft2(load_image)))
return spectrum

def extract_spectrum_on_frame(frame):
spectrum = np.abs(np.fft.fftshift(np.fft.fft2(frame)))
return spectrum
def spectrum_to_see(picture):
spectrum = np.fft.fftshift(np.fft.fft2(picture))
# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(spectrum)
# Handle zero values
small_value = 1e-10 # You can adjust this value as needed,
it should be small enough not to affect the overall spectrum
significantly
magnitude_spectrum[magnitude_spectrum == 0] =
small_value
# Apply logarithmic transformation
magnitude_spectrum = 20 * np.log(magnitude_spectrum)
magnitude_spectrum[np.isinf(magnitude_spectrum)] =
0 # Replace -inf values with 0

magnitude_spectrum[np.isnan(magnitude_spectrum)] =
0 # Replace nan values with 0
magnitude_spectrum =
cv2.normalize(magnitude_spectrum, None, 0, 255,
cv2.NORM_MINMAX, cv2.CV_8U)
# magnitude_spectrum_normalized =
(magnitude_spectrum - np.min(magnitude_spectrum)) /
(np.max(magnitude_spectrum) -
np.min(magnitude_spectrum))
return magnitude_spectrum

def compare_spectra(spectrum1, spectrum2):
correlation = np.corrcoef(spectrum1.flatten(),
spectrum2.flatten())[0, 1]
return correlation
if __name__ == '__main__':
image1 = ... # Đường dẫn đến hình ảnh thứ nhất
image2 = ... # Đường dẫn đến hình ảnh thứ hai
spectrum1 = extract_spectrum(image1)
spectrum2 = extract_spectrum(image2)
correlation = compare_spectra(spectrum1, spectrum2)
print("Correlation:", correlation)