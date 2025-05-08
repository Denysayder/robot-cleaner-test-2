import cv2
def histogram_equalization(image_path):
# Đọc ảnh từ đường dẫn
image = cv2.imread(image_path)
#thiết lập output
output_image_path =
'image/histogram_equalized_result.jpg'
# Chuyển ảnh sang ảnh xám
gray_image = cv2.cvtColor(image,
cv2.COLOR_BGR2GRAY)
# Cân bằng histogram
equalized_image = cv2.equalizeHist(gray_image)
# color_equalized_image =
cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
cv2.imwrite(output_image_path, equalized_image)
# Trả về ảnh đã được cân bằng histogram

return output_image_path
def histogram_equalization_on_frame(frame):
# Chuyển ảnh sang ảnh xám
gray_frame = cv2.cvtColor(frame,
cv2.COLOR_BGR2GRAY)
# Cân bằng histogram
equalized_frame = cv2.equalizeHist(gray_frame)
# color_equalized_frame = cv2.cvtColor(equalized_frame,
cv2.COLOR_GRAY2BGR)
# Trả về ảnh đã được cân bằng histogram
return equalized_frame