import cv2 as cv
import numpy as np
def process_video_perspective(input_file):
output_file = "video/view_result.mp4"
multiplier_width = 0.246
multiplier_height = 0.1875
# Open the video file
cap = cv.VideoCapture(input_file)
# Get the video's frame rate and dimensions
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# Define the transformation function
def getTransform(points):
pts1 = np.float32(points)
pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
return cv.getPerspectiveTransform(pts1, pts2)
# Create video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')
output = cv.VideoWriter(output_file, fourcc, fps, (300,
300))
while True:
# Read the next frame from the video
ret, frame = cap.read()
if not ret:
# End of video
break
# Copy the frame
img_ori = frame.copy()

# Define the points for transformation
points = np.array([[width * multiplier_width, height *
multiplier_height],
[width * (1 - multiplier_width), height *
multiplier_height],
[width * (1 - multiplier_width), height * (1 -
multiplier_height)],
[width * multiplier_width, height * (1 -
multiplier_height)]], dtype=np.float32)
m = getTransform(points)
# Apply perspective transformation
dst = cv.warpPerspective(np.float32(img_ori), m, (300,
300))
dst = np.array(dst, dtype='uint8')
# Write the processed frame to the output video
output.write(dst)
# Release the video file and writer
cap.release()
output.release()
# Close any open windows
cv.destroyAllWindows()
return output_file
def process_image_perspective(input_file):
output_file = "image/view_result.jpg"
multiplier_width = 0.246
multiplier_height = 0.1875
# Load the image
img = cv.imread(input_file)
# Define the points for transformation
height, width = img.shape[:2]
points = np.array([[width * multiplier_width, height *
multiplier_height],
[width * (1 - multiplier_width), height *
multiplier_height],
[width * (1 - multiplier_width), height * (1 -
multiplier_height)],
[width * multiplier_width, height * (1 -
multiplier_height)]], dtype=np.float32)
# Define the transformation function
def getTransform(points):
pts1 = np.float32(points)
pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

return cv.getPerspectiveTransform(pts1, pts2)
m = getTransform(points)
# Apply perspective transformation to the image
dst = cv.warpPerspective(img, m, (300, 300))
# Save the processed image
cv.imwrite(output_file, dst)
return output_file
def perform_realtime_perspective_transform(frame, width,
height):
# Define the points for perspective transformation
multiplier_width = 0.246
multiplier_height = 0.1875
points = np.array([[width * multiplier_width, height *
multiplier_height],
[width * (1 - multiplier_width), height *
multiplier_height],
[width * (1 - multiplier_width), height * (1 -
multiplier_height)],
[width * multiplier_width, height * (1 -
multiplier_height)]], dtype=np.float32)
# Define the transformation function
def get_transform(points):
pts1 = np.float32(points)
pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
return cv.getPerspectiveTransform(pts1, pts2)
m = get_transform(points)
transformed_frame = cv.warpPerspective(frame, m, (300,
300))
return transformed_frame