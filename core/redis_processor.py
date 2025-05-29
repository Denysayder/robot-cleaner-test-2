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