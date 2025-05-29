import redis
import cv2
import numpy as np
import base64
from core.arduino_sender import send_to_arduino, \
    open_serial_connection, send_data, close_serial_connection, \
    find_port

import termios
import sys
import tty
import config.config_secret as config_secret
import os

RedisHost = config_secret.RedisHost
RedisPort = config_secret.RedisPort
RedisPassword = config_secret.RedisPassword
user_id = os.getenv("USER_ID") or "demo"
key = lambda name: f"user:{user_id}:{name}"

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def connect_redis(RedisHost, RedisPort, RedisPassword):
    r = redis.Redis(RedisHost, RedisPort, db=0, password=RedisPassword, ssl=True, decode_responses=True)
    return r

def send_signal(r, redis_key, value):
    if r.ping():
        r.set(redis_key, value)
        print(f"Tín hiệu đã được gửi thành công đến key: {key(name)}.")
    else:
        print("Không có kết nối đến Redis. Tín hiệu không được gửi.")

def receive_signal(r, redis_key):      # ← имя параметра теперь очевидно
    value = r.get(redis_key)           # ① НЕ оборачиваем его снова
    if value is not None:
        print(f"Received data from key: {redis_key}")
    else:
        print(f"Nothing from key: {redis_key}")
    return value

def disconnect_redis(r):
    r.close()

def receive_realtime_signal_and_send_to_arduino(redis_keys):
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, 9600)
    while True:
        received_values = []
        for key in redis_keys:
            received_value = receive_signal(redis_conn, key)
            received_values.append(received_value)
        for i, value in enumerate(received_values):
            send_data(ser, value, i + 1)
    close_serial_connection(ser)
    disconnect_redis(redis_conn)

def receive_realtime_frame():
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    full_frame_data = ""
    frame_for_decode = ""
    while True:
        char = getch()
        if char == 'q':
            break
        frame_chunk = receive_signal(redis_conn, key("video"))
        parts = frame_chunk.split("_")
        chunk_prefix = int(parts[0])
        data = parts[1]
        chunk_suffix = parts[2]
        full_frame_data += data
        if chunk_suffix == "endframe":
            frame_for_decode = full_frame_data
            full_frame_data = ""
            video_frame = decode_frame(frame_for_decode)
            if video_frame is not None:
                cv2.imshow("Video", video_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    disconnect_redis(redis_conn)

def decode_frame(frame_for_decode):
    img_bytes = base64.b64decode(frame_for_decode)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


import serial
import serial.tools.list_ports
import time
import sys
import termios
import tty

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def find_port():
    arduino_port = None
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print("Port:", port.device)
        print("Description:", port.description)
        print()
        if 'Arduino' in port.description or 'ACM' in port.description:
            arduino_port = port.device
            print("Arduino is connected to", arduino_port)
            break
    if arduino_port is None:
        print("Arduino is not connected to any port")
    return arduino_port

def open_serial_connection(arduino_port, baud_rate):
    if arduino_port is None:
        return None
    try:
        ser = serial.Serial(
            arduino_port,
            baud_rate,
            exclusive=True,
            timeout=0.005,
            dsrdtr=False,  # <-- предотвращает сброс Arduino
            rtscts=False
        )
        time.sleep(2)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print("An error occurred while opening the serial connection:", e)
        return None

def send_data(ser, baudrate, data, channel=0):
    if ser is None:
        print("Connection to Arduino is not established yet.")
        return
    try:
        if isinstance(data, bytes):
            data_bytes = data
        else:
            if not isinstance(data, str):
                data_str = str(data)
            else:
                data_str = data
            data_bytes = data_str.encode()
        data_with_newline = str(channel).encode() + str(":").encode() + data_bytes + b'#'
        ser.write(data_with_newline)
        print(f"Data: {data} has been sent to channel {channel}")
    except serial.SerialException as e:
        print("An error occurred while sending data:", e)

def receive_data(ser):
    if ser is None:
        print("Connection is not established yet.")
        return []
    received_data = []
    while ser.in_waiting:
        line = ser.readline().decode('latin-1').strip()
        parts = line.split(":")
        if len(parts) == 2:
            prefix, data = parts[0], parts[1]
            if prefix.isdigit():  # Kiểm tra tính hợp lệ của tiền tố
                prefix = int(prefix)
                if len(received_data) > prefix:
                    received_data[prefix] = data
                else:
                    received_data.extend([None] * (prefix - len(received_data)))
                    received_data.append(data)
    if len(received_data) > 0:
        print("Đã nhận được tín hiệu từ Arduino")
    else:
        print("Không có tín hiệu từ Arduino")
    return received_data

def close_serial_connection(ser):
    if ser is not None:
        ser.close()

def send_to_arduino(data, baud_rate, channel=1):
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)
    send_data(ser, baud_rate, data, channel)
    close_serial_connection(ser)

def receive_from_arduino(baud_rate):
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)
    received_commands = receive_data(ser)
    close_serial_connection(ser)
    return received_commands

def command_to_send_to_arduino(float_value, base_data):
    if float_value < 0:
        return "G"
    else:
        return base_data


import pandas as pd
import numpy as np

def save_parameters_to_csv(parameter_file_path, mean, std):
    parameters_df = pd.DataFrame({
        'Parameter': ['mean', 'std'],
        'Values': [mean, std]
    })
    parameters_df.to_csv(parameter_file_path, index=False)

def estimate_gaussian_parameters(file_path, Parameter_file_path, feature_name='Values'):
    data = pd.read_csv(file_path)
    feature_values = data[feature_name].values
    mean = np.mean(feature_values)
    std = np.std(feature_values)
    save_parameters_to_csv(Parameter_file_path, mean, std)

def get_parameters(parameter_file_path):
    data = pd.read_csv(parameter_file_path)
    mean = data[data['Parameter'] == 'mean']['Values'].values[0]
    std = data[data['Parameter'] == 'std']['Values'].values[0]
    return mean, std

def predict_group(value, mean_clean, std_clean, mean_dirty, std_dirty):
    p_clean = 1 / (np.sqrt(2 * np.pi) * std_clean) * np.exp(-0.5 * ((value - mean_clean) / std_clean) ** 2)
    p_dirty = 1 / (np.sqrt(2 * np.pi) * std_dirty) * np.exp(-0.5 * ((value - mean_dirty) / std_dirty) ** 2)
    if p_clean > p_dirty:
        return "Clean"
    else:
        return "G"

if __name__ == '__main__':
    new_value = 0.67
    clean_file = 'data/processed/data_100_150.csv'
    dirty_file = 'data/processed/data_100_150_dirty.csv'
    clean_parameters_file = 'data/raw/clean_parameters.csv'
    dirty_parameters_file = 'data/raw/dirty_parameters.csv'

    estimate_gaussian_parameters(clean_file, clean_parameters_file)
    estimate_gaussian_parameters(dirty_file, dirty_parameters_file)

    mean_clean, std_clean = get_parameters('data/raw/clean_parameters.csv')
    mean_dirty, std_dirty = get_parameters('data/raw/dirty_parameters.csv')

    predicted_group = predict_group(new_value, mean_clean, std_clean, mean_dirty, std_dirty)
    print("Number {} is belong to group : {}".format(new_value, predicted_group))


import skimage
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from scipy.stats import entropy
import numpy as np

def compare_images(frame1, frame2):
    if frame1 is None or frame2 is None:
        return 0
    size = (300, 300)
    image1 = cv.resize(frame1, size)
    image2 = cv.resize(frame2, size)
    if (np.max(image1) - np.min(image1)) == 0:
        image1_normalized = image1
    else:
        image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))

    if (np.max(image2) - np.min(image2)) == 0:
        image2_normalized = image2
    else:
        image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    image1_bw = np.where(image1_normalized >= 0.5, 1, 0)
    image2_bw = np.where(image2_normalized >= 0.5, 1, 0)
    similarity = ssim(image1_bw, image2_bw, data_range=1.0)
    se = np.mean((image1_bw - image2_bw) ** 2)
    mse = se**(1 / 2)
    histogram1 = np.histogram(image1_normalized, bins=256)[0]
    histogram2 = np.histogram(image2_normalized, bins=256)[0]
    joint_histogram = np.histogram2d(image1_normalized.flatten(),
                                     image2_normalized.flatten(), bins=256)[0]
    nmi = ((entropy(histogram1) + entropy(histogram2)) / entropy(joint_histogram.flatten()))
    combined_score = (nmi * similarity * (1 - mse))**(1 / 3)
    if np.isnan(combined_score):
        combined_score = 0
    return combined_score



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

if __main__ == '__main__':
    frame = cv2.imread('input_frame.jpg')
    threshold_value = 200
    result_frame = perform_brightness_thresholding(frame, threshold_value)
    cv2.imshow('Result Frame', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


import cv2 as cv
import numpy as np
import time
import base64
import csv
import os
import glob
import random
from core.arduino_sender import open_serial_connection, close_serial_connection, send_data, command_to_send_to_arduino, find_port, receive_data
from core.redis_processor import connect_redis, receive_signal, disconnect_redis, send_signal
from processing.histogram_equalizer import histogram_equalization, histogram_equalization_on_frame
from processing.adjust_brightness import adjust_brightness_on_frame, adjust_brightness_on_image
from processing.correlation import extract_spectrum, extract_spectrum_on_frame, spectrum_to_see
from processing.similarity_NMI import compare_images
from processing.light_to_dark import perform_brightness_thresholding, perform_brightness_thresholding_on_image
from ml.Bayes_class_decision import predict_group, get_parameters
from serial import serial_for_url
import config.config_secret as config_secret

user_id = os.getenv("USER_ID") or "demo"
key = lambda name: f"user:{user_id}:{name}"

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


def Main_Run(user_id, reference_image, baud_rate, redis_receive_keys, arduino_data_keys):
    RedisHost = config_secret.RedisHost
    RedisPort = config_secret.RedisPort
    RedisPassword = config_secret.RedisPassword
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    key = lambda name: f"user:{user_id}:{name}"
    ps = redis_conn.pubsub(ignore_subscribe_messages=True)
    ps.subscribe(key("robot:commands"))


    ARDUINO_URL = 'rfc2217://localhost:4000'
    try:
        ser = serial_for_url(ARDUINO_URL, baudrate=baud_rate, timeout=0.005)
        print(f"[Arduino] connected via {ARDUINO_URL}")
    except Exception as exc:
        print(f"[Arduino] connection failed: {exc}")
        ser = None

    def poll_arduino_and_store():
        """Раз в цикл читаем строку CSV и кладём значения в Redis."""
        if ser is None:
            return
        raw = ser.readline().decode(errors='ignore').strip()
        if not raw:
            return
        try:
            work, temp, s1, s2, s3, s4, water, batt, moved = raw.split(',')
        except ValueError:
            print(f"[Arduino] bad line: {raw}")
            return
        redis_conn.hset(key('telemetry'),mapping={
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

    load_reference_image = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
    reference_fourier_frame = spectrum_to_see(load_reference_image)
    cv.imwrite("image/fourier_image.jpg", reference_fourier_frame)

    def process_video_from_folder(video_folder_path, ser):
        video_playlist = sorted(glob.glob(os.path.join(video_folder_path, '*.mp4')))
        random.shuffle(video_playlist)
        current_video_idx = 0
        current_video_frames = []
        current_frame_idx   = 0

        image_files = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')))

        if not image_files:
            print(f"No image files found in {video_folder_path}")
            return

        width = 480
        height = 240

        kalman = cv.KalmanFilter(1, 1, 0)
        kalman.transitionMatrix = np.array([[1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1]], dtype=np.float32)
        kalman.processNoiseCov = np.array([[1e-5]], dtype=np.float32)
        kalman.measurementNoiseCov = np.array([[1e-3]], dtype=np.float32)
        kalman.errorCovPost = np.array([[1]], dtype=np.float32)
        kalman.statePost = np.array([[0]], dtype=np.float32)

        frame_rate_limit = 10
        frame_interval = 1 / frame_rate_limit
        last_frame_time = time.time() - frame_interval
        timestamps = []
        values = []
        first_frame_time = time.time()
        mean_clean, std_clean = get_parameters('data/raw/clean_parameters.csv')
        mean_dirty, std_dirty = get_parameters('data/raw/dirty_parameters.csv')

        csvfile = open('data/processed/data_test.csv', 'w', newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamps', 'Values'])
        while True:
            msg = ps.get_message()
            if msg and msg["type"] == "message":
                cmd = msg["data"]           # строка, т.к. decode_responses=True
                if cmd == "start_clean":
                    redis_conn.set(key("cleaning_control"), "start")
                    redis_conn.set(key("camera"), "on")
                elif cmd == "stop_clean":
                    redis_conn.set(key("cleaning_control"), "stop")
                    redis_conn.set(key("camera"), "off")
                elif cmd == "status":
                    # при желании отправьте статус обратно, например:
                    redis_conn.set(key("robot:state"), "running")

            cleaning_control = receive_signal(redis_conn, key("cleaning_control"))  # 'start' | 'stop' | None
            if cleaning_control == 'stop':
                time.sleep(0.1)
                continue
            if cleaning_control != 'start':
                time.sleep(0.1)
                continue

            if not current_video_frames:
                video_file_path = video_playlist[current_video_idx]
                extract_frames_from_video(video_file_path, video_folder_path)
                current_video_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')))
                current_frame_idx = 0
                if not current_video_frames:
                    current_video_idx = (current_video_idx + 1) % len(video_playlist)
                    continue

            image_path = current_video_frames[current_frame_idx]
            current_frame_idx += 1

            cleaning_control = receive_signal(redis_conn, key('cleaning_control'))
            if cleaning_control == 'stop':
                break
            current_time = time.time()
            poll_arduino_and_store()
            elapsed_time = current_time - last_frame_time
            if elapsed_time >= frame_interval:
                last_frame_time = current_time
                frame = cv.imread(image_path)

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv.resize(frame, (width, height))

                camera_control = receive_signal(redis_conn, key("camera"))
                if camera_control == "on":
                    _, img_encoded = cv.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    img_base64 = base64.b64encode(img_bytes).decode("ascii")
                    redis_conn.set(key("video"), f"1_{img_base64}_endframe")

                    brightness_adjusted_frame = adjust_brightness_on_frame(frame, 100)

                    perform_brightness_thresholding_frame = perform_brightness_thresholding(brightness_adjusted_frame, 150)

                    histograme_equalized_frame = histogram_equalization_on_frame(perform_brightness_thresholding_frame)

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
                    for redis_field in redis_receive_keys:
                        received_value = receive_signal(redis_conn, redis_field)
                        received_values.append(received_value)

                    for i, value in enumerate(received_values):
                        send_data(ser, baud_rate, value, i)

                    TheCommand = predict_group(NMI_Score_filtered, mean_clean, std_clean, mean_dirty, std_dirty)
                    send_data(ser, baud_rate, TheCommand, 1)
                    redis_conn.hset(key('telemetry'), mapping={'panelStatus': TheCommand})
                    print(f"NMI_Score: {NMI_Score_filtered:.2f}")
                    telemetry = redis_conn.hgetall(key('telemetry'))
                    print(f"T={telemetry.get('temperature')}°C "
                        f"Batt={telemetry.get('battery')}% "
                        f"H₂O={telemetry.get('water')}%")

                    kalman.statePost = kalman_corrected
                    print(f"TheCommand: {TheCommand}")
                if current_frame_idx >= len(current_video_frames):
                    current_video_idx   = (current_video_idx + 1) % len(video_playlist)
                    current_video_frames = []
                    current_frame_idx    = 0
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
    Main_Run(user_id, histogram_equalized_file, baud_rate, redis_keys, arduino_data_keys)


#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
}

void loop() {
  static uint32_t sec = 0;
  static float temperature = 25.0;
  static int   water       = 100;
  static float battery     = 100.0;  // Изменили на float для плавности

  /*---- динамика температуры чуть «мягче»  ----*/
  temperature = constrain(temperature + random(-10, 11) * 0.04, 20, 60);

  /*---- солнечная мощность с небольшим шумом  ----*/
  float daylight = max(0.0f, sin(sec * 0.0007f));            // плавное "солнце"
  int   s1 = (int)(300 + daylight * 900) + random(-15, 16);  // 280-1220 W
  int   s2 = 30 + (sec % 20);                                // 30→49 за 20 с
  int   s3 = 400 + (sec * 20) % 400;                         // 400→800
  int   s4 = 5 + (sec * 4) % 90;                             // 5→95

  /*---- расход ресурсов/накопление пути  ----*/
  if (sec != 0 && sec % 3 == 0 && water > 90) water--;      // −5 % за 20 с

  // Плавное снижение батареи с небольшим шумом
  if (battery > 5.0) {
    float batteryDecline = 0.4 + random(-2, 3) * 0.05;      // ~0.4% в секунду ± шум
    battery = max(5.0f, battery - batteryDecline);
  }

  long moved = sec * 7;      // расстояние растёт ~линейно

  Serial.print(sec);            Serial.print(',');
  Serial.print(temperature, 1); Serial.print(',');
  Serial.print(s1);             Serial.print(',');
  Serial.print(s2);             Serial.print(',');
  Serial.print(s3);             Serial.print(',');
  Serial.print(s4);             Serial.print(',');
  Serial.print(water);          Serial.print(',');
  Serial.print((int)battery);   Serial.print(',');          // Приводим к int для вывода
  Serial.println(moved);

  sec++;
  delay(1000);                  // 1 сек → 20 строк за демонстрацию
}