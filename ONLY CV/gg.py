import cv2
import torch
import numpy as np
import pyautogui
import time
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
from grabscreen import grab_screen

# Load the pre-trained model and processor
model_name = "huggingface/lane-detection"  # Replace with actual model name
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageSegmentation.from_pretrained(model_name)

# Function to process the image and perform lane detection
def process_img(image):
    # Pre-process the image and convert it into a format that the model understands
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processing (e.g., extracting lane segmentation mask)
    logits = outputs.logits
    lane_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # You can also extract other information like lane coordinates for further processing if needed
    # For now, we just use the lane mask for control
    return lane_mask, image

# Placeholder functions for steering control
def right():
    print("Turn Right")

def left():
    print("Turn Left")

def straight():
    print("Go Straight")

# Main loop to continuously grab screen and process lane detection
last_time = time.time()
while True:
    # Capture the screen (you can adjust the region as needed)
    screen = grab_screen(region=(0, 40, 800, 640))
    print('Frame took {} seconds'.format(time.time() - last_time))
    last_time = time.time()

    # Process the captured image for lane detection
    lane_mask, original_image = process_img(screen)

    # Optional: Visualize the lane detection result
    cv2.imshow('Lane Mask', lane_mask * 255)  # Show lane mask as a binary image
    cv2.imshow('Original Image', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Show original image

    # Use lane information (m1, m2 are placeholders for lane-related features)
    m1, m2 = np.mean(lane_mask, axis=1), np.mean(lane_mask, axis=0)

    if m1 < 0 and m2 < 0:
        right()
    elif m1 > 0 and m2 > 0:
        left()
    else:
        straight()

    # Close the program if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
