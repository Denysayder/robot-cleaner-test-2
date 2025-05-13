import cv2 as cv
import numpy as np
import time
import base64
import csv
import os
import glob
# from fourier_processors import perform_fourier_transform_and_compare_to_reference_fourier_image, process_image_fourier
from arduino_sender import open_serial_connection, close_serial_connection, send_data, command_to_send_to_arduino, find_port, receive_data
from perspective_processor import perform_realtime_perspective_transform, process_image_perspective
from redis_processor import connect_redis, receive_signal, disconnect_redis, send_signal
from histogram_equalizer import histogram_equalization, histogram_equalization_on_frame
# from remove_reflection import remove_reflection, remove_reflection_on_frame
from adjust_brightness import adjust_brightness_on_frame, adjust_brightness_on_image
from correlation import extract_spectrum, extract_spectrum_on_frame, spectrum_to_see
from similarity_NMI import compare_images
# from remove_reflection import remove_reflection_on_frame, remove_reflection
from light_to_dark import perform_brightness_thresholding, perform_brightness_thresholding_on_image
from Bayes_class_decision import predict_group, get_parameters
from serial import serial_for_url
import config_secret

def extract_frames_from_video(video_path, output_folder):
    cap = cv.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv.imwrite(output_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


# Main Run: Nhận tín hiệu từ Redis, biến đổi fourier và so sánh từng frame lấy từ Camera với ảnh sạch, rồi chuyển lệnh đến Arduino
def Main_Run(reference_image, baud_rate, redis_receive_keys, arduino_data_keys):
    # Open Redis Connection
    RedisHost = config_secret.RedisHost
    RedisPort = config_secret.RedisPort
    RedisPassword = config_secret.RedisPassword
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)

    # Open Serial connection
    # receive_data=""
    # arduino_port = find_port()
    # ser = open_serial_connection(arduino_port, baud_rate)
        # --- Arduino in Wokwi: TCP‑UART на 4000 ---
    ARDUINO_URL = 'rfc2217://localhost:4000'  # порт задан в wokwi.toml
    try:
        ser = serial_for_url(ARDUINO_URL, baudrate=baud_rate, timeout=0.005)
        print(f"[Arduino] connected via {ARDUINO_URL}")
    except Exception as exc:
        print(f"[Arduino] connection failed: {exc}")
        ser = None
            # ---------- Читаем CSV‑телеметрию ----------

    def poll_arduino_and_store():
        """Раз в цикл читаем строку CSV и кладём значения в Redis."""
        if ser is None:
            return
        raw = ser.readline().decode(errors='ignore').strip()
        if not raw:
            return  # таймаут
        try:
            work, temp, s1, s2, s3, s4, water, batt, moved = raw.split(',')
        except ValueError:
            print(f"[Arduino] bad line: {raw}")
            return
        redis_conn.hset('telemetry', mapping={
            'workTime'   : work,
            'temperature': temp,
            'sensor1'    : s1,
            'sensor2'    : s2,
            'sensor3'    : s3,
            'sensor4'    : s4,
            'water'      : water,
            'battery'    : batt,
            'moved'      : moved
        })

    # Load the reference image for comparison
    load_reference_image = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
    reference_fourier_frame = spectrum_to_see(load_reference_image)
    cv.imwrite("image/fourier_image.jpg", reference_fourier_frame)

    # Turn on module camera raspberry
    # Set video resolution
    def process_video_from_folder(video_folder_path, ser):
        # Get list of image files in the folder (assuming they are sorted by name)
        video_file_path = os.path.join(video_folder_path, "input_video.mp4")
        extract_frames_from_video(video_file_path, video_folder_path)

        image_files = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')))

        if not image_files:
            print(f"No image files found in {video_folder_path}")
            return

        # Set the same resolution as in the original code
        width = 480
        height = 240

        # Initialize variables for Kalman Filter
        kalman = cv.KalmanFilter(1, 1, 0)
        kalman.transitionMatrix = np.array([[1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1]], dtype=np.float32)
        kalman.processNoiseCov = np.array([[1e-5]], dtype=np.float32)
        kalman.measurementNoiseCov = np.array([[1e-3]], dtype=np.float32)
        kalman.errorCovPost = np.array([[1]], dtype=np.float32)
        kalman.statePost = np.array([[0]], dtype=np.float32)

        frame_rate_limit = 10  # Giới hạn số khung hình mỗi giây
        frame_interval = 1 / frame_rate_limit
        last_frame_time = time.time() - frame_interval
        timestamps = []
        values = []
        first_frame_time = time.time()
        mean_clean, std_clean = get_parameters('clean_parameters.csv')
        mean_dirty, std_dirty = get_parameters('dirty_parameters.csv')

        csvfile = open('data_test.csv', 'w', newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamps', 'Values'])
        # Loop through image files instead of capturing from camera
        for image_path in image_files:
            current_time = time.time()
            poll_arduino_and_store()
            elapsed_time = current_time - last_frame_time
            if elapsed_time >= frame_interval:
                last_frame_time = current_time
                # Read frame from file instead of camera
                frame = cv.imread(image_path)

                # Resize frame to match original resolution if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv.resize(frame, (width, height))

                camera_control = receive_signal(redis_conn, "camera")
                if camera_control == "on":
                    _, img_encoded = cv.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    img_base64 = base64.b64encode(img_bytes)
                    chunk_size = 60000
                    chunks = [img_base64[i:i + chunk_size] for i in range(0, len(img_base64), chunk_size)]
                    flagged_chunks = [f"{i + 1}_{chunk}" for i, chunk in enumerate(chunks)]
                    flagged_chunks[-1] = f"{flagged_chunks[-1]}_endframe"
                    flagged_chunks = [f"{chunk}_notyet" if i != len(flagged_chunks) - 1 else chunk for i, chunk in enumerate(flagged_chunks)]
                    for flagged_chunk in flagged_chunks:
                        redis_conn.set('video', flagged_chunk)

                    # Apply brightness adjustment
                    brightness_adjusted_frame = adjust_brightness_on_frame(frame, 100)

                    # Turn the part too white to black
                    perform_brightness_thresholding_frame = perform_brightness_thresholding(brightness_adjusted_frame, 150)

                    # Apply histogram equalizer to the frame
                    histograme_equalized_frame = histogram_equalization_on_frame(perform_brightness_thresholding_frame)

                    # Apply extract_spectrum
                    fourier_frame = spectrum_to_see(histograme_equalized_frame)

                    NMI_Score = compare_images(reference_fourier_frame, fourier_frame)

                    kalman_prediction = kalman.predict()
                    kalman_corrected = kalman.correct(np.array([[NMI_Score]], dtype=np.float32))
                    NMI_Score_filtered = kalman_corrected[0, 0]

                    if not isinstance(values, list):
                        values = [values]
                    timestamp = current_time - first_frame_time
                    csvwriter.writerow([timestamp, NMI_Score_filtered])

                    received_values = []
                    for key in redis_receive_keys:
                        received_value = receive_signal(redis_conn, key)
                        received_values.append(received_value)

                    for i, value in enumerate(received_values):
                        send_data(ser, baud_rate, value, i)

                    TheCommand = predict_group(NMI_Score_filtered, mean_clean, std_clean, mean_dirty, std_dirty)
                    ser = send_data(ser, baud_rate, TheCommand, 1)

                    print(f"NMI_Score: {NMI_Score_filtered:.2f}")
                    telemetry = redis_conn.hgetall('telemetry')
                    print(f"T={telemetry.get('temperature')}°C "
                          f"Batt={telemetry.get('battery')}% "
                          f"H₂O={telemetry.get('water')}%")

                    kalman.statePost = kalman_corrected
                    print(f"TheCommand: {TheCommand}")
                processing_time = time.time() - current_time
                sleep_time = max(0, frame_interval - processing_time)
                time.sleep(sleep_time)

        csvfile.close()
        close_serial_connection(ser)
        disconnect_redis(redis_conn)

    process_video_from_folder("video_frames", ser)

if __name__ == '__main__':
    redis_keys = ["move"]
    arduino_data_keys = ["Temperature", "Sensor1", "Sensor2", "Sensor3", "Sensor4", "water", "battery", "Moved Distance"]
    input_image_file = 'image/webcam_image5.jpg'

    brightness_adjusted_file = adjust_brightness_on_image(input_image_file, 100)
    brightness_threshold_file = perform_brightness_thresholding_on_image(brightness_adjusted_file, 150)
    histogram_equalized_file = histogram_equalization(brightness_threshold_file)

    baud_rate = 9600
    Main_Run(histogram_equalized_file, baud_rate, redis_keys, arduino_data_keys)
