import cv2

def histogram_equalization(image_path):
    image = cv2.imread(image_path)
    output_image_path = 'image/histogram_equalized_result.jpg'
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    cv2.imwrite(output_image_path, equalized_image)
    return output_image_path

def histogram_equalization_on_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    return equalized_frame