import cv2
import torch
import pytesseract
import os
import glob
import re
import time
import pandas as pd

# Start the timer
start_time = time.time()

# Define the function to check if a box is inside an ROI
def is_inside_roi(box, roi):
    (startX, startY, endX, endY) = box
    (roi_startX, roi_startY, roi_endX, roi_endY) = roi
    return startX >= roi_startX and startY >= roi_startY and endX <= roi_endX and endY <= roi_endY

# Define a function to preprocess the image for better OCR results
def preprocess_image_for_ocr(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_img = cv2.resize(thresh_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return resized_img

# Define a function to extract and filter bus numbers using regular expressions
def extract_bus_numbers(text):
    pattern = r'\b\d{3}[A-Za-z]?\b'
    matches = re.findall(pattern, text)
    return matches

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the folder path containing the images
image_folder_path = '/home/leshane/Downloads/frames'
image_files = sorted(glob.glob(os.path.join(image_folder_path, '*.jpg')))

# Define the ROIs
bus_status_roi = (440, 253, 1278, 495)
signage_roi = (160, 210, 375, 302)
bus_number_roi = (443, 261, 843, 329)
queue_rois = [(0, 334, 1294, 693)]

# Initialize a list to keep track of queue counts over time
queue_count_history = []

# Initialize variables to store the final results
final_signage_texts = []
final_bus_numbers = []

# Open text files to save the results
with open('busNum.txt', 'w') as bus_file, open('queue.txt', 'w') as queue_file:
    for image_path in image_files:
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        # Perform inference
        results = model(frame)

        # Parse results
        predictions = results.pred[0]

        # Initialize a counter for people in the queue
        people_in_queue = 0

        # Check for person detection and count in the queue ROI
        for *box, conf, cls in predictions:
            if cls == 0:  # Assuming class 0 is 'person'
                if is_inside_roi(box, queue_rois[0]):
                    people_in_queue += 1

        # Preprocess and extract text for bus status, signage, and bus number
        signage_img = preprocess_image_for_ocr(frame[signage_roi[1]:signage_roi[3], signage_roi[0]:signage_roi[2]])
        signage_text = pytesseract.image_to_string(signage_img, config='--psm 6').strip()
        bus_numbers = extract_bus_numbers(signage_text)

        # If three sets of numbers are detected, add them to the final results
        if len(bus_numbers) == 3:
            final_signage_texts.append(bus_numbers)

        # Preprocess and extract text for the bus number
        bus_number_img = preprocess_image_for_ocr(frame[bus_number_roi[1]:bus_number_roi[3], bus_number_roi[0]:bus_number_roi[2]])
        bus_number_text = pytesseract.image_to_string(bus_number_img, config='--psm 6').strip()
        bus_number = extract_bus_numbers(bus_number_text)

        # If a bus number is detected, add it to the final results
        if bus_number:
            final_bus_numbers.append(bus_number)

        # Write the extracted information to the busNum.txt file
        bus_file.write(f'Image: {os.path.basename(image_path)}\n')
        bus_file.write(f'Signage Text: {signage_text}\n')
        bus_file.write(f'Bus Numbers: {bus_number}\n\n')

        # Write the queue count history to the queue.txt file
        queue_file.write(f'Image: {os.path.basename(image_path)}\n')
        queue_file.write(f'Queue Count History: {people_in_queue}\n\n')

    # After processing all images, write the final results to the busNum.txt file
    bus_file.write(f'Final Signage Texts: {final_signage_texts}\n')
    bus_file.write(f'Final Bus Numbers: {final_bus_numbers}\n')
