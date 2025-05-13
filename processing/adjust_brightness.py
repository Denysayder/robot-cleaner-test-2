import cv2
import numpy as np

def adjust_brightness_on_frame(frame, target_brightness):
    avg_brightness = np.mean(frame)
    adjustment_ratio = target_brightness / avg_brightness
    adjusted = np.clip(frame, None, 150)
    adjusted = cv2.convertScaleAbs(frame, alpha=adjustment_ratio)
    return adjusted

def adjust_brightness_on_image(image_file, target_brightness):
    output_file = "image/brightness_adjusted_image.jpg"
    image = cv2.imread(image_file)
    adjusted = adjust_brightness_on_frame(image, target_brightness)
    cv2.imwrite(output_file, adjusted)
    return output_file

def main():
    cap = cv2.VideoCapture(0)
    target_brightness = 100
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            print("Không thể đọc khung hình.")
        adjusted_frame = adjust_brightness_on_frame(frame, target_brightness)
        avg_brightness_original = np.mean(frame)
        avg_brightness_adjusted = np.mean(adjusted_frame)
        cv2.putText(frame, f'Original: {avg_brightness_original}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(adjusted_frame, f'Adjusted: {avg_brightness_adjusted}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Original', frame)
        cv2.imshow('Adjusted', adjusted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
