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
    magnitude_spectrum = np.abs(spectrum)
    small_value = 1e-10
    magnitude_spectrum[magnitude_spectrum == 0] = small_value
    magnitude_spectrum = 20 * np.log(magnitude_spectrum)
    magnitude_spectrum[np.isinf(magnitude_spectrum)] = 0
    magnitude_spectrum[np.isnan(magnitude_spectrum)] = 0
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return magnitude_spectrum

def compare_spectra(spectrum1, spectrum2):
    correlation = np.corrcoef(spectrum1.flatten(), spectrum2.flatten())[0, 1]
    return correlation

if __name__ == '__main__':
    image1 = ...
    image2 = ...
    spectrum1 = extract_spectrum(image1)
    spectrum2 = extract_spectrum(image2)
    correlation = compare_spectra(spectrum1, spectrum2)
    print("Correlation:", correlation)
