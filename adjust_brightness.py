import cv2
import numpy as np

def adjust_brightness_on_frame(frame, target_brightness):
    # Tính độ sáng trung bình của hình ảnh
    avg_brightness = np.mean(frame)
    # Tính tỷ lệ điều chỉnh dựa trên độ sáng trung bình hiện tại
    # và độ sáng mục tiêu
    adjustment_ratio = target_brightness / avg_brightness
    # # Giới hạn độ sáng lớn nhất của mỗi pixel là 150
    adjusted = np.clip(frame, None, 150)
    # Điều chỉnh độ sáng của hình ảnh mà không thay đổi kiểu
    # dữ liệu
    adjusted = cv2.convertScaleAbs(frame, alpha=adjustment_ratio)
    return adjusted

def adjust_brightness_on_image(image_file, target_brightness):
    output_file = "image/brightness_adjusted_image.jpg"
    image = cv2.imread(image_file)
    adjusted = adjust_brightness_on_frame(image, target_brightness)
    cv2.imwrite(output_file, adjusted)
    return output_file

def main():
    # Mở webcam
    cap = cv2.VideoCapture(0)
    target_brightness = 100
    # Kiểm tra xem webcam đã được mở chưa
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    while True:
        # Đọc khung hình từ webcam
        ret, frame = cap.read()
        if not ret:
            break
            print("Không thể đọc khung hình.")
        # Điều chỉnh độ sáng
        adjusted_frame = adjust_brightness_on_frame(frame, target_brightness)
        # Tính độ sáng trung bình của hai hình ảnh
        avg_brightness_original = np.mean(frame)
        avg_brightness_adjusted = np.mean(adjusted_frame)
        # Hiển thị độ sáng trung bình lên trên hình ảnh
        cv2.putText(frame, f'Original: {avg_brightness_original}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(adjusted_frame, f'Adjusted: {avg_brightness_adjusted}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Hiển thị hình ảnh gốc và hình ảnh đã được điều chỉnh
        # độ sáng
        cv2.imshow('Original', frame)
        cv2.imshow('Adjusted', adjusted_frame)
        # Chờ nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
