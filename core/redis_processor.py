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

RedisHost = config_secret.RedisHost
RedisPort = config_secret.RedisPort
RedisPassword = config_secret.RedisPassword

# hàm bấm nút để ngừng
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# mở cổng kết nối với redis
def connect_redis(RedisHost, RedisPort, RedisPassword):
    r = redis.Redis(RedisHost, RedisPort, db=0, password=RedisPassword, ssl=True, decode_responses=True)
    return r

# gởi tín hiệu đến redis
def send_signal(r, key, value):
    if r.ping():
        r.set(key, value)
        print(f"Tín hiệu đã được gửi thành công đến key: {key}.")
    else:
        print("Không có kết nối đến Redis. Tín hiệu không được gửi.")

# nhận mảng tín hiệu từ redis
def receive_signal(r, key):
    value = r.get(key)
    if value is not None:
        print(f"Received data from key: {key}")
        return value
    else:
        print(f"Nothing from key: {key}")
        return None

def disconnect_redis(r):
    r.close()

def receive_realtime_signal_and_send_to_arduino(redis_keys):
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, 9600)
    while True:
        # if msvcrt.kbhit() and msvcrt.getch() == b'q':
        #     break
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
        # Thoát đọc khi bấm q từ bàn phím
        char = getch()
        if char == 'q':
            break
        # Đọc tín hiệu từ key movie
        frame_chunk = receive_signal(redis_conn, "video")
        # tách tín hiệu thành các phần
        parts = frame_chunk.split("_")
        chunk_prefix = int(parts[0])
        data = parts[1]
        chunk_suffix = parts[2]
        # Nối các phần data lại theo thứ tự (chưa đúng lắm)
        full_frame_data += data
        # Reset lại cho vòng lặp mới
        if chunk_suffix == "endframe":
            frame_for_decode = full_frame_data
            full_frame_data = ""
            video_frame = decode_frame(frame_for_decode)
            if video_frame is not None:
                cv2.imshow("Video", video_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Exit loop if 'q' key is pressed
                    break
    cv2.destroyAllWindows()
    disconnect_redis(redis_conn)

def decode_frame(frame_for_decode):
    # decode frame_for_decode from base64 to bytes
    # Chuyển đổi img_base64 thành chuỗi bytes
    img_bytes = base64.b64decode(frame_for_decode)
    # Chuyển đổi chuỗi bytes thành mảng numpy
    nparr = np.frombuffer(img_bytes, np.uint8)
    # Đọc mảng numpy thành frame
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# receive_realtime_frame()
# # Nhận tín hiệu theo thời gian thực từ key "move" & "toggle"
# redis_keys = ["move", "toggle"]
# receive_realtime_signal_and_send_to_arduino(redis_keys)
# # Kết nối tới Redis
# redis_host = 'redis-12030.c73.us-east-1-2.ec2.cloud.redislabs.com'  # Địa chỉ Redis server
# redis_port = 12030  # Cổng Redis
# redis_password = 'ZYtsQ1tAG4osutc294eHcQLRFKdxKbE3'  # Mật khẩu Redis (nếu có)
# redis_conn = connect_redis(redis_host, redis_port, redis_password)
# # # Lấy hình ảnh từ key 'movie' và giải mã
# test_string = receive_signal(redis_conn, "video")
# print(test_string)
# disconnect_redis(redis_conn)
