import cv2
import numpy as np

def remove_reflection(image_path, threshold_adder):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    # thiết lập output
    output_image_path = 'image/reflection_remove_result.jpg'
    # Chuyển đổi ảnh sang không gian màu LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Tính toán ngưỡng dựa trên giá trị sáng trung bình của ảnh
    mean_brightness = np.mean(lab_image[:, :, 0])
    threshold = mean_brightness + threshold_adder  # Có thể điều chỉnh giá trị ngưỡng tùy ý
    # Xác định vùng phản chiếu bằng cách tìm các điểm sáng hơn ngưỡng
    reflection_mask = cv2.threshold(lab_image[:, :, 0], threshold, 255, cv2.THRESH_BINARY)[1]
    reflection_mask = cv2.dilate(reflection_mask, None, iterations=2)  # Mở rộng vùng phản chiếu
    # Tạo hình chữ nhật ước lượng chứa vùng phản chiếu
    contours, _ = cv2.findContours(reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Cắt bỏ phần phản chiếu khỏi ảnh gốc
    processed_image = image.copy()
    processed_image[y:y+h, x:x+w] = cv2.medianBlur(processed_image[y:y+h, x:x+w], 15)
    # Lưu ảnh đã được xử lý
    cv2.imwrite(output_image_path, processed_image)
    # Trả về ảnh đã xử lý
    return output_image_path

def remove_reflection_on_frame(frame, threshold_adder):
    # Chuyển đổi ảnh sang không gian màu BGR nếu ảnh đầu vào là ảnh grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Chuyển đổi ảnh sang không gian màu LAB
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # Tính toán ngưỡng dựa trên giá trị sáng trung bình của ảnh
    mean_brightness = np.mean(lab_frame[:, :, 0])
    threshold = mean_brightness + threshold_adder  # Có thể điều chỉnh giá trị ngưỡng tùy ý
    # Xác định vùng phản chiếu bằng cách tìm các điểm sáng hơn ngưỡng
    reflection_mask = cv2.threshold(lab_frame[:, :, 0], threshold, 255, cv2.THRESH_BINARY)[1]
    reflection_mask = cv2.dilate(reflection_mask, None, iterations=2)  # Mở rộng vùng phản chiếu
    # Tạo hình chữ nhật ước lượng chứa vùng phản chiếu
    contours, _ = cv2.findContours(reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Check if contours is not empty
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Cắt bỏ phần phản chiếu khỏi ảnh gốc
        processed_frame = frame.copy()
        processed_frame[y:y+h, x:x+w] = cv2.medianBlur(processed_frame[y:y+h, x:x+w], 15)
    else:
        processed_frame = frame.copy()  # Assign a copy of the input frame when contours is empty
    # Trả về ảnh đã xử lý
    return processed_frame
