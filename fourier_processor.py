import cv2 as cv
import numpy as np

def process_image_fourier(input_file):
    output_file = "image/fourier_result.jpg"
    # Load the input image
    img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    assert img is not None, "Image file could not be read"
    # Perform Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # Convert the magnitude spectrum to uint8 for writing to image
    magnitude_spectrum_uint8 = cv.normalize(magnitude_spectrum, None, 0, 255,
                                            cv.NORM_MINMAX, cv.CV_8U)
    # Save the result image
    cv.imwrite(output_file, magnitude_spectrum_uint8)
    return output_file

def process_video_fourier(input_file):
    output_file = "video/fourier_result.mp4"
    # Open the input video
    cap = cv.VideoCapture(input_file)
    # Check if the video file was successfully opened
    assert cap.isOpened(), "Video file could not be opened"
    # Get the video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Create a VideoWriter object for the output video
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_file, fourcc, fps, (width, height), False)
    # Process each frame in the video
    while True:
        ret, frame = cap.read()
        # Check if there are no more frames to read
        if not ret:
            break
        # Convert the frame to grayscale
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Perform Fourier transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # Convert the magnitude spectrum to uint8 for writing to video
        magnitude_spectrum_uint8 = cv.normalize(magnitude_spectrum, None, 0, 255,
                                                cv.NORM_MINMAX, cv.CV_8U)
        # Write the frame to the output video
        out.write(magnitude_spectrum_uint8)
    # Release the video file and the output video
    cap.release()
    out.release()
    return output_file

def perform_fourier_transform_and_compare_to_reference_fourier_image(frame, reference_image):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Perform Fourier transform
    dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # Normalize the magnitude spectrum for display
    magnitude_spectrum = cv.normalize(magnitude_spectrum, None, 0, 255,
                                      cv.NORM_MINMAX, dtype=cv.CV_8U)
    # Compare the magnitude spectrum with the reference image
    diff = cv.absdiff(reference_image, magnitude_spectrum)
    diff_percentage = np.mean(diff) / 255 * 100
    # Convert back to BGR for display
    result = cv.cvtColor(magnitude_spectrum, cv.COLOR_GRAY2BGR)
    return result, diff_percentage
