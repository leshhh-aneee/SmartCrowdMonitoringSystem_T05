import cv2
import torch
import pytesseract
import os
import glob
import re
import time
import math
import pandas as pd
import csv  
import numpy as np
from datetime import datetime
import collections

from Prediction import Predict

class BusQueueCounter:
    def __init__(self, image_folder_path, device_count_path, bus_info_path, wav_file, model_file, scaler_file, features_csv_file):
        self.image_folder_path = image_folder_path
        self.device_count_path = device_count_path
        self.bus_info_path = bus_info_path
        self.wav_file = wav_file
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.features_csv_file = features_csv_file
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.last_detected_bus_number = []
        self.consecutive_no_count = 0
        self.consecutive_no_threshold = 5

    def is_inside_roi(self, box, roi):
        (startX, startY, endX, endY) = box
        (roi_startX, roi_startY, roi_endX, roi_endY) = roi
        return startX >= roi_startX and startY >= roi_startY and endX <= roi_endX and endY <= roi_endY

    def preprocess_image_for_ocr(self, img):
        resized_img = cv2.resize(img, None, fx=2.7, fy=2.4, interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img_dilated = cv2.dilate(img_gray, kernel, iterations=1)
        img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
        _, img_threshold = cv2.threshold(cv2.medianBlur(img_eroded, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_threshold

    def extract_bus_numbers(self, text):
        pattern = r'\b\d{3}[A-Za-z]?\b'
        matches = re.findall(pattern, text)
        return matches

    def extract_signage_bus_numbers(self, frame, roi):
        signage_img = self.preprocess_image_for_ocr(frame[roi[1]:roi[3], roi[0]:roi[2]])
        signage_text = pytesseract.image_to_string(signage_img, config='--psm 6').strip()
        bus_numbers = self.extract_bus_numbers(signage_text)
        return bus_numbers

    def count_queue(self, predictions, roi):
        people_in_queue = 0
        for *box, conf, cls in predictions:
            if cls == 0:  # Class 0 is 'person'
                if self.is_inside_roi(box, roi):
                    people_in_queue += 1
        return people_in_queue

    def check_bus_number_on_bus(self, frame, roi):
        bus_number_img = self.preprocess_image_for_ocr(frame[roi[1]:roi[3], roi[0]:roi[2]])
        bus_number_text = pytesseract.image_to_string(bus_number_img, config='--psm 6').strip()
        bus_number = self.extract_bus_numbers(bus_number_text)
        return bus_number

    def is_bus_present(self, predictions, roi):
        for *box, conf, cls in predictions:
            if self.is_inside_roi(box, roi):
                return True
        return False

    def update_queue_count(self, image_path):
        latest_count_df = pd.read_csv(self.device_count_path)
        bus_info_df = pd.read_csv(self.bus_info_path)

        current_image_index = bus_info_df[bus_info_df['Image'] == os.path.basename(image_path)].index.item()

        try:
            current_latest_count_row = latest_count_df.iloc[current_image_index]

            queue_count = bus_info_df.at[current_image_index, 'Queue Count']
            total_count = current_latest_count_row['Total Count']
            average_queue_count = math.ceil((queue_count + total_count) / 2)

            bus_info_df.at[current_image_index, 'Queue Count'] = average_queue_count

            bus_info_df.to_csv(self.bus_info_path, index=False)
        except IndexError:
            print("Error: Image index is out of bounds. Skipping...")

    def initialize_bus_numbers(self, signage_roi):
        start_time = time.time()
        all_bus_numbers = []
        image_files = sorted(glob.glob(os.path.join(self.image_folder_path, '*.jpg')))
        for image_path in image_files:
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            bus_numbers = self.extract_signage_bus_numbers(frame, signage_roi)
            all_bus_numbers.extend(bus_numbers)

        bus_number_counts = collections.Counter(all_bus_numbers)
        top_bus_numbers = [number for number, count in bus_number_counts.most_common(3)]
        print(f"Time taken: {time.time() - start_time} seconds")
        return top_bus_numbers

    def write_initial_bus_numbers_to_csv(self, top_bus_numbers):
        with open(self.bus_info_path, mode='w', newline='') as bus_file:
            csv_writer = csv.writer(bus_file)
            csv_writer.writerow(['Image', 'Bus Arrived', 'Signage Text', 'Bus Numbers', 'Queue Count'])
            csv_writer.writerow(['', '', ', '.join(top_bus_numbers), '', ''])

    def run_count_queue(self, frame, predictions, roi, bus_status_roi, bus_number_roi, image_path, top_bus_numbers, frame_idx):
        start_time = time.time()
        people_in_queue = self.count_queue(predictions, roi)
        bus_present = self.is_bus_present(predictions, bus_status_roi)

        if bus_present:
            if not self.last_detected_bus_number:
                bus_number = self.check_bus_number_on_bus(frame, bus_number_roi)
                self.last_detected_bus_number = bus_number
            else:
                bus_number = self.last_detected_bus_number
        else:
            bus_number = []
            self.consecutive_no_count += 1
            # If bus of status is no for 5 or more frames, do predictions
            if self.consecutive_no_count >= self.consecutive_no_threshold: # consecutive_no_threshold is 5
                # Calculate the corresponding time in the WAV file based on frame index
                wav_duration = self.get_wav_duration()
                frame_duration = 1  # Assuming 1 frame equals 1 second
                start_time_sec = frame_idx * frame_duration
                end_time_sec = start_time_sec + 5  # Predict for the next 5 seconds
                if end_time_sec <= wav_duration:
                    # Trigger prediction for the last 5 seconds of the WAV file
                    self.trigger_prediction(start_time_sec, end_time_sec)

        with open(self.bus_info_path, mode='a', newline='') as bus_file:
            csv_writer = csv.writer(bus_file)
            csv_writer.writerow([os.path.basename(image_path), 'Yes' if bus_present else 'No', ', '.join(top_bus_numbers), ', '.join(bus_number), people_in_queue])
            print(f"Time taken: {time.time() - start_time} seconds")

        self.update_queue_count(image_path)

    def trigger_prediction(self, start_time_sec, end_time_sec):
        # Initialize the Predict class and process the WAV file for the specified time range
        predictor = Predict(self.wav_file, self.model_file, self.scaler_file, self.features_csv_file, update_interval=3, export_csv=False)
        predictor.process_and_plot(start_time_sec, end_time_sec)

    def get_wav_duration(self):
        # Get the duration of the WAV file in seconds
        try:
            import wave
            with wave.open(self.wav_file, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"Error getting WAV duration: {e}")
            return None

    def process_frames(self, signage_roi, bus_status_roi, bus_number_roi, queue_rois):
        top_bus_numbers = self.initialize_bus_numbers(signage_roi)
        self.write_initial_bus_numbers_to_csv(top_bus_numbers)

        image_files = sorted(glob.glob(os.path.join(self.image_folder_path, '*.jpg')))
        for frame_idx, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            results = self.model(frame)
            predictions = results.pred[0]
            self.run_count_queue(frame, predictions, queue_rois, bus_status_roi, bus_number_roi, image_path, top_bus_numbers, frame_idx)

