import cv2 as cv
import numpy as np
import time
import base64
import picamera
import csv
#from fourier_processors import
perform_fourier_transform_and_compare_to_reference_fou
rier_image, process_image_fourier
from arduino_sender import open_serial_connection,
close_serial_connection, send_data,
command_to_send_to_arduino, find_port, receive_data
from perspective_processor import
perform_realtime_perspective_transform,
process_image_perspective
from redis_processors import connect_redis, receive_signal,
disconnect_redis, send_signal
from histogram_equalizer import histogram_equalization,
histogram_equalization_on_frame
# from remove_reflection import remove_reflection,
remove_reflection_on_frame
from adjust_brightness import adjust_brightness_on_frame,
adjust_brightness_on_image
from correlation import extract_spectrum,
extract_spectrum_on_frame, spectrum_to_see


from similarity_NMI import compare_images
# from remove_reflection import
remove_reflection_on_frame, remove_reflection
from light_to_dark import perform_brightness_thresholding,
perform_brightness_thresholding_on_image
from Bayes_class_decision import predict_group,
get_parameters
#Main Run: Nhận tín hiệu từ Redis, biến đổi fourier và so
sánh từng frame lấy từ Camera với ảnh sạch, rồi chuyển lệnh
đến Arduino
def Main_Run(reference_image, baud_rate,
redis_receive_keys, arduino_data_keys):
#Open Redis Connection
RedisHost =
'redis-17060.c299.asia-northeast1-1.gce.cloud.redislabs.com'
RedisPort = 17060
RedisPassword =
'YS9EKyvuTG5Q3HGxKFem48ePa7ykaV93'
redis_conn = connect_redis(RedisHost , RedisPort,
RedisPassword)
# Open Serial connection
# receive_data=""
arduino_port = find_port()
ser = open_serial_connection(arduino_port, baud_rate)
# Load the reference image for comparison
load_reference_image = cv.imread(reference_image,
cv.IMREAD_GRAYSCALE)
reference_fourier_frame =
spectrum_to_see(load_reference_image)
cv.imwrite("image/fourier_image.jpg",
reference_fourier_frame)
# Open the camera for video capture
#cap = cv.VideoCapture(0) # Use 0 for the default camera,
or specify the camera index
# Get the camera's frame rate and dimensions
#fps = cap.get(cv.CAP_PROP_FPS)
#width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#height =
int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#turn on module camera raspberry
#set video resolution
with picamera.PiCamera() as camera:
camera.resolution = (480,240)
camera.framerate = 10
#start the video preview

camera.start_preview()
width = camera.resolution[0]
height = camera.resolution[1]
# Initialize variables for Kalman Filter
kalman = cv.KalmanFilter(1, 1, 0)
kalman.transitionMatrix = np.array([[1]],
dtype=np.float32)
kalman.measurementMatrix = np.array([[1]],
dtype=np.float32)
kalman.processNoiseCov = np.array([[1e-5]],
dtype=np.float32)
kalman.measurementNoiseCov = np.array([[1e-3]],
dtype=np.float32)
kalman.errorCovPost = np.array([[1]],
dtype=np.float32)
kalman.statePost = np.array([[0]], dtype=np.float32)
frame_rate_limit = 10 # Giới hạn số khung hình mỗi
giây
frame_interval = 1 / frame_rate_limit
last_frame_time = time.time()
timestamps= []
values=[]
first_frame_time=time.time()
mean_clean, std_clean =
get_parameters('clean_parameters.csv')
mean_dirty, std_dirty =
get_parameters('dirty_parameters.csv')
#count = 0
csvfile = open('data_test.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(['Timestamps', 'Values'])
while True:
# Read the next frame from the camera
#ret, frame = cap.read()
#if not ret:
# Error in capturing frame
# break
current_time = time.time()
elapsed_time = current_time - last_frame_time
if elapsed_time >= frame_interval:
last_frame_time = current_time


stream
=np.empty((camera.resolution[1]*camera.resolution[0]*3),
dtype=np.uint8)
camera.capture(stream, 'bgr')
frame =stream.reshape ((camera.resolution[1],
camera.resolution[0], 3))
camera_control = receive_signal(redis_conn,
"camera")
if camera_control == "on":
# Chuyển đổi frame thành dạng nhị phân
_, img_encoded = cv.imencode('.jpg', frame)
img_bytes = img_encoded.tobytes()
# Chuyển đổi img_bytes thành chuỗi base64
img_base64 =
base64.b64encode(img_bytes).decode('utf-8')
chunk_size = 60000 # Độ dài của mỗi phần chuỗi
chunks = [img_base64[i:i+chunk_size] for i in
range(0, len(img_base64), chunk_size)]
# Gắn thêm flag để xác định thứ tự của các chuỗi
mới
flagged_chunks = [f"{i+1}_{chunk}" for i, chunk
in enumerate(chunks)] # example chunk: "[1_data 2_info"
flagged_chunks[-1] =
f"{flagged_chunks[-1]}_endframe"
# Thêm hậu tố "_notyet" cho các phần tử, trừ phần
tử cuối cùng
flagged_chunks = [f"{chunk}_notyet" if i !=
len(flagged_chunks) - 1 else chunk for i, chunk in
enumerate(flagged_chunks)]
# Gửi frame đến Redis chỉ khi đã trôi qua đủ
khoảng thời gian
for flagged_chunk in flagged_chunks:
redis_conn.set('video', flagged_chunk)

#Receive data from Arduino
# received_datas = receive_data(ser)
# print("Thông tin nhận được:")
# for prefix, data in enumerate(received_data):
# if data is not None:

# command_function =
arduino_data_keys[prefix] if prefix <
len(CommandFunction) else "Unknown Function"
# print(f"Command {prefix}: {data} -
Function: {command_function}")
# Apply perspective transformation to the frame
#transformed_frame =
perform_realtime_perspective_transform(frame, width,
height)
# # Turn the part too white to black
# perform_brightness_thresholding_frame=
perform_brightness_thresholding(transformed_frame, 100)
# Apply brightness adjustment
brightness_adjusted_frame =
adjust_brightness_on_frame(frame, 100)
# Turn the part too white to black
perform_brightness_thresholding_frame=
perform_brightness_thresholding(brightness_adjusted_fram
e, 150)


#Apply histogram equalizer to the frame
histograme_equalized_frame =
histogram_equalization_on_frame(perform_brightness_thres
holding_frame)
#Apply reflection remove to the frame
# reflection_removed_frame =
remove_reflection_on_frame(histograme_equalized_frame,
50)
#Apply extract_spectrum
fourier_frame =
spectrum_to_see(histograme_equalized_frame)
#if count != -1
# count += 1
# if count == 50:
# cv.imwrite("image/fourier_frame.jpg",
fourier_frame)


# count = -1
NMI_Score=
compare_images(reference_fourier_frame,fourier_frame)
# fourier_frame_shape = fourier_frame.shape
# reference_fourier_frame_shape =
reference_fourier_frame.shape
# cv.putText(fourier_frame, f"size:
{fourier_frame_shape}", (10,30),
cv.FONT_HERSHEY_SIMPLEX, 1,
# (0, 255, 0), 2)
# cv.putText(reference_fourier_frame, f"size:
{reference_fourier_frame_shape}", (10,30),
cv.FONT_HERSHEY_SIMPLEX, 1,
# (0, 255, 0), 2)
#Nmi comparison to get the difference
# cv.imshow("Fourier Frame", fourier_frame)
# cv.imshow("brightness",
brightness_adjusted_frame)
# cv.imshow("Histogram",
histograme_equalized_frame)
# cv.imshow("Reference Frame",
reference_fourier_frame)

# Apply Kalman Filter to diff_percentage
kalman_prediction = kalman.predict()
kalman_corrected =
kalman.correct(np.array([[NMI_Score]], dtype=np.float32))
NMI_Score_filtered = kalman_corrected[0, 0]
#list of time and values
if not isinstance(values, list):
values = [values]
timestamp = current_time-first_frame_time
csvwriter.writerow([timestamp,
NMI_Score_filtered])
#timestamps.append(timestamp)
#values.append(NMI_Score_filtered)
# Receive data from Redis
received_values = []
for key in redis_receive_keys:
received_value = receive_signal(redis_conn, key)
received_values.append(received_value)


# Send the diff_percentage_filtered to
Arduino
for i, value in enumerate(received_values):
send_data(ser,baud_rate ,value, i)
TheCommand =
predict_group(NMI_Score_filtered, mean_clean, std_clean,
mean_dirty, std_dirty)
ser = send_data(ser, baud_rate, TheCommand, 1)
# Display the frame and difference percentage in
real-time
# cv.putText(frame, f"NMI Score:
{NMI_Score_filtered:.2f}", (10, 30),
cv.FONT_HERSHEY_SIMPLEX, 1,
# (0, 255, 0), 2)
# cv.imshow('Real-time Comparison', frame)
# Send datas from Arduino to Redis
#for prefix, data in enumerate(received_datas):
# if data is not None:
# send_signal(redis_conn,
arduino_data_keys[prefix], data)
print(f"NMI_Score: {NMI_Score_filtered:.2f}")
kalman.statePost = kalman_corrected
print(f"TheCommand: {TheCommand}")


# Release the camera
camera.stop_preview()
camera.close()
# Close connection to Arduino
close_serial_connection(ser)
disconnect_redis(redis_conn)
# Close the 'Real-time Comparison' window
# cv.destroyAllWindows()
if __name__ == '__main__':
redis_keys = ["move"]
arduino_data_keys = ["Temperature", "Sensor1",
"Sensor2", "Sensor3", "Sensor4", "water", "battery", "Moved
Distance"]
input_image_file = 'image/webcam_image5.jpg'
#view_image_file =
process_image_perspective(input_image_file)
#remove_reflection_file =
remove_reflection(input_image_file, 50)


brightness_adjusted_file =
adjust_brightness_on_image(input_image_file, 100)
brightness_threshold_file =
perform_brightness_thresholding_on_image(brightness_adj
usted_file, 150)
histogram_equalized_file =
histogram_equalization(brightness_threshold_file)
baud_rate = 9600
Main_Run(histogram_equalized_file, baud_rate,
redis_keys, arduino_data_keys)