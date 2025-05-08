#!/bin/bash

# Обновление пакетов
sudo apt update

# Установка системных зависимостей
sudo apt install -y python3-pip python3-opencv libatlas-base-dev libjpeg-dev libtiff-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libcanberra-gtk* libhdf5-dev libhdf5-serial-dev libhdf5-103 \
    libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 \
    libopenblas-dev liblapack-dev libatlas-base-dev gfortran \
    libusb-1.0-0-dev

# Установка Python-библиотек
pip3 install --upgrade pip
pip3 install opencv-python numpy pandas pyserial redis scikit-image scipy

# Проверка установки
echo "Установленные версии:"
python3 -c "import cv2; print('cv2:', cv2.__version__)"
python3 -c "import numpy as np; print('numpy:', np.__version__)"
python3 -c "import pandas as pd; print('pandas:', pd.__version__)"
python3 -c "import serial; print('pyserial:', serial.__version__)"
python3 -c "import redis; print('redis:', redis.__version__)"
python3 -c "import skimage; print('skimage:', skimage.__version__)"
python3 -c "import scipy; print('scipy:', scipy.__version__)"
