Я пишу диплом на тему "Інформаційна система моніторингу та автоматизованого очищення сонячних панелей роботизованими пристроями". Із минулих чатів ти знаєш що включає мій диплом. Мені треба написати записку, перші два розділи якої складається з наступних пунктів:

ВСТУП
1. АНАЛІЗ ПРЕДМЕТНОЇ ОБЛАСТІ ТА ПОСТАНОВКА ЗАДАЧ
1.1 Актуальність проблеми
1.2
1.2 Аналіз існуючих технічних і програмних рішень (показати 2-3 рішення)
1.2.1 Ecoppia T4
1.2.2 SolarCleano F1
1.2.3 BladeRanger Pleco
1.4 Формулювання вимог до інформаційної системи
Висновки до розділу 1

2. Опис структури системи (до 15 сторінок)
2.1 Вибір архітектури системи (кресленик)
2.2 Вибір мови програмування
	2.2.1 JavaScript
	2.2.2 C#
	2.2.3 Python
2.3 Вибір фреймворку
2.3.1 Django
2.3.2 FastAPI
2.3.3 Flask
2.4 Вибір бази даних
	2.4.1 PostgreSQL
	2.4.2 MongoDB
	2.4.3 MySQL
2.5 Redis
2.6 Wokwi
2.7 Вибір середовища розробки
2.8 Інформаційне забезпечення
	2.8.1 Вхідні дані
	2.8.2 Вихідні дані
Висновки до розділу 2

3. Реалізація системи
3.1 Архітектура та середовище розгортання
3.2 Алгоритм оцінки ступеня забруднення
3.2.1 Попередня обробка (корекція яскравості, фільтр Калмана)
3.2.2 Формування спектра Фур’є та обчислення NMI
3.2.3 Байєсівська класифікація «Clean / Dirty»
3.3 Backend-сервер і API
3.4 Алгоритми аналізу та управління
3.5 Клієнтський інтерфейс
Висновки до розділу 3



Тому розпиши ТІЛЬКИ І ТІЛЬКИ пункт "3.2.1 Попередня обробка (корекція яскравості, фільтр Калмана) ". Не використовуй таблиці, бажано тільки текст. Обсяг 300 слів! Якщо раніше ми все загально описували, то тут вже дуже конкретно треба розписати що саме конкретно реалізовано! Нижче ти можеш знайти код в якому реалізовано алгоритми, але я від тебе хочу не опису кожної строки коду, а опису самого алгоритму і як він реалізується.

adjust_brightness.py
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

realtime_comparison.py:
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