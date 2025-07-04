import cv2
import numpy as np
import time
from ultralytics import YOLO
from grabscreen import grab_screen
# Load your YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Replace with your model path

# Initialize timing
last_time = time.time()

while True:
    # Capture the screen (adjust region as needed)
    screen = grab_screen(region=(0, 40, 800, 640))
    print('Frame took {:.3f} seconds'.format(time.time() - last_time))
    last_time = time.time()

    # Convert BGR (from grab) to RGB for YOLO
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # YOLO inference
    results = model(screen_rgb)

    # Plot result (with lane mask)
    annotated_frame = results[0].plot()

    # Show YOLO-detected frame
    cv2.imshow('YOLO Lane Detection', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
