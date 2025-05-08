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

# 'USB-SERIAL CH-340'
def find_port():
    arduino_port = None
    # Tìm kiếm cổng kết nối
    ports = serial.tools.list_ports.comports()
    # Lặp qua danh sách các cổng
    for port in ports:
        # List available ports and their descriptions
        print("Port:", port.device)
        print("Description:", port.description)
        print()
        # Kiểm tra tên và mô tả cổng
        if 'Arduino' or 'ACM' in port.description:
            # Xác định cổng kết nối của Arduino
            arduino_port = port.device
            print("Arduino is connected to", arduino_port)
            break
    # Nếu không tìm thấy cổng kết nối Arduino
    if arduino_port is None:
        print("Arduino is not connected to any port")
    return arduino_port

def open_serial_connection(arduino_port, baud_rate):
    if arduino_port is None:
        return None
    try:
        ser = serial.Serial(arduino_port, baud_rate, exclusive=True)
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print("An error occurred while opening the serial connection:", e)
        return None

def send_data(ser, baudrate, data, channel=0):
    if ser is None:
        print("Connection to Arduino is not established yet.")
        arduino_port = find_port()
        ser = open_serial_connection(arduino_port, baudrate)
        return ser
    try:
        # Check if data is already in bytes format
        if isinstance(data, bytes):
            data_bytes = data
        else:
            # Convert the data to string format if it's not already a string
            if not isinstance(data, str):
                data_str = str(data)
            else:
                data_str = data
            # Convert the data to bytes format
            data_bytes = data_str.encode()
        # Convert the channel to bytes format and add newline character to the data
        data_with_newline = str(channel).encode() + str(":").encode() + data_bytes + b'#'
        # Send data to Arduino
        ser.write(data_with_newline)
        print(f"Data: {data} has been sent to channel {channel}")
        return ser
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
    # Tìm Port kết nối
    arduino_port = find_port()
    # Open serial connection to Arduino
    ser = open_serial_connection(arduino_port, baud_rate)
    # Send data to Arduino
    send_data(ser, data, channel)
    # Close serial connection to Arduino
    close_serial_connection(ser)

def receive_from_arduino(baud_rate):
    # Tìm Port kết nối
    arduino_port = find_port()
    # Open serial connection to Arduino
    ser = open_serial_connection(arduino_port, baud_rate)
    # Receive data from Arduino
    received_commands = receive_data(ser)
    # Close serial connection to Arduino
    close_serial_connection(ser)
    return received_commands

def command_to_send_to_arduino(float_value, base_data):
    if float_value < 0:
        return "G"
    else:
        return base_data

# float_value = 7
# data = command_to_send_to_arduino(float_value)
# send_to_arduino(data, 9600)
# # Gọi hàm để nhận dữ liệu từ Arduino
# received_commands = receive_from_arduino(9600)
# # In ra các tín hiệu nhận được
# for command in received_commands:
#     print("Received command:", command)

## Test Nhận tín hiệu từ Arduino
# Kết nối đến Serial và khởi tạo ser
# arduino_port = find_port()
# baud_rate = 115200
# ser = open_serial_connection(arduino_port, baud_rate)
# CommandFunction = ["Work Time", "Temperature",
#     "Front-Left Distance Sensor", "Front-Right Distance Sensor",
#     "Back-Left Distance Sensor", "Back-Right Distance Sensor", "Left Over Water", "Battery Left",
#     "Moved Distance", "Over Work"]
# received_data = receive_data(ser)

# print("Thông tin nhận được:")
# for prefix, data in enumerate(received_data):
#     if data is not None:
#         command_function = CommandFunction[prefix] if prefix < len(CommandFunction) else "Unknown Function"
#         print(f"Command {prefix}: {data} - Function: {command_function}")
# close_serial_connection(ser)
