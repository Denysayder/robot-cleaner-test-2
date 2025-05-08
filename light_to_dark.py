import numpy as np
import cv2
import __main__

def perform_brightness_thresholding(frame, brightness_threshold):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    value = np.where(value > brightness_threshold, 0, value)
    hsv[:, :, 2] = value
    processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return processed_frame

def perform_brightness_thresholding_on_image(image_file, brightness_threshold):
    output_file = "image/brightness_threshold_image.jpg"
    image = cv2.imread(image_file)
    processed_image = perform_brightness_thresholding(image, brightness_threshold)
    cv2.imwrite(output_file, processed_image)
    return output_file

# Example usage:
if __main__ == '__main__':
    frame = cv2.imread('input_frame.jpg')  # Replace 'input_frame.jpg' with the path to your input frame
    threshold_value = 200
    result_frame = perform_brightness_thresholding(frame, threshold_value)
    cv2.imshow('Result Frame', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
