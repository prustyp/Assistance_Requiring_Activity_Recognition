import numpy as np
import cv2

import time
capture_duration = 5
cap = cv2.VideoCapture(0) # Capture video from camera
fps=60
cap.set(cv2.CAP_PROP_FPS, fps)
# Get the width and height of frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('/Users/pprusty05/google_drive/Deep_Learning/video1.mp4', fourcc, 20.0, (width, height))

start_time = time.time()
while( int(time.time() - start_time) < capture_duration ):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
    else:
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()