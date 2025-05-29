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
