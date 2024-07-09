import cv2
import numpy as np
import os
from datetime import datetime

# Set the DISPLAY environment variable to use the monitor connected to the Raspberry Pi
os.environ['DISPLAY'] = ':0'

# Function to control the ACT LED
def set_led(state):
    try:
        with open('/sys/class/leds/ACT/brightness', 'w') as led_file:  # Update this path based on your system
            led_file.write('1' if state else '0')
    except PermissionError:
        print("Permission denied: Could not write to LED control file. Please run the script with sudo.")

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create directory to save images
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

try:
    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # Compute the absolute difference between frames
        diff = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

        if motion_detected:
            # Turn on the ACT LED
            set_led(True)
            # Capture and save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(save_dir, f"motion_{timestamp}.jpg"), frame2)
        else:
            # Turn off the ACT LED
            set_led(False)

        # Display the resulting frame
        cv2.imshow('Motion Detection', frame2)

        # Update the previous frame
        gray1 = gray2.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    # Turn off the ACT LED
    set_led(False)
