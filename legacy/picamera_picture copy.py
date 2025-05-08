import picamera
import cv2
import picamera.array
import numpy as np
#Hiên thị hình ảnh từ camera trên màn hình
def show_camera_preview():
cv2.namedWindow("Camera Preview",
cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Preview", 480,240)
#biến để lưu ảnh
frame= None
with picamera.PiCamera() as camera:
camera.resolution = (480,240)
camera.framerate = 24
Automated and Self Powered Solar Panel Cleaning Robot
camera.start_preview()

while True:
stream
=np.empty((camera.resolution[1]*camera.resolution[0]*3),
dtype=np.uint8)
camera.capture(stream, 'bgr')
frame =stream.reshape ((camera.resolution[1],
camera.resolution[0], 3))
#show picture to the screen
cv2.imshow("Camera Preview",frame)
#wait key
key= cv2.waitKey(1)
if key == ord ('q'):
image_path="/home/snetel/Desktop/GammaImage
Process/image/webcam_image5.jpg"
cv2.imwrite(image_path, frame)
print("already taken picture: ",image_path)
break
camera.stop_preview()
camera.close
cv2.destroyAllWindows()
if __name__== '__main__':
show_camera_preview()