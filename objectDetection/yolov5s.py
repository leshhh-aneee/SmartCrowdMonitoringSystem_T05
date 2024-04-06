import cv2
import torch
import pytesseract
from pytesseract import Output

# Define the function to check if a box is inside an ROI
def is_inside_roi(box, roi):
    (startX, startY, endX, endY) = box
    (roi_startX, roi_startY, roi_endX, roi_endY) = roi
    return startX >= roi_startX and startY >= roi_startY and endX <= roi_endX and endY <= roi_endY

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video file or stream
video_path = '/home/leshane/Downloads/finalVideo.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit() 

bus_status_roi = (496, 389, 1278, 495)
signage_roi = (160, 210, 375, 302)
bus_number_roi = (418, 274, 843, 329)

# Define the ROIs for the queues (x1, y1, x2, y2)
queue_rois = [
    (0, 334, 1294, 693),  # ROI for the left queue
    # Add additional ROIs for other queues if necessary
]

# Initialize a list to keep track of queue counts over time
queue_count_history = []

# Define a variable to store the bus status
# bus_status = 'bus status: have not arrived'

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Perform inference
    results = model(frame)

    # Parse results
    predictions = results.pred[0]

    # Initialize a flag to check if a bus is detected within the ROI
    bus_detected = False

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in predictions:
        if cls == 5:  
            bus_detected = True
            
    
    # Extract the bus numbers from the signage
    signage_img = frame[signage_roi[1]:signage_roi[3], signage_roi[0]:signage_roi[2]]
    signage_text = pytesseract.image_to_string(signage_img, config='--psm 6').strip()

    # Extract the bus number from the bus
    bus_number_img = frame[bus_number_roi[1]:bus_number_roi[3], bus_number_roi[0]:bus_number_roi[2]]
    bus_number_text = pytesseract.image_to_string(bus_number_img, config='--psm 6').strip()

    # Check if the bus number matches any of the numbers from the signage
    bus_numbers_from_signage = signage_text.split()
    bus_arrived = bus_number_text in bus_numbers_from_signage
    
    # Update the bus status based on detection
    if bus_detected==True:
        bus_status = f'bus status: arrived, number: {bus_number_text}'
    else:
        bus_status = 'bus status: have not arrived'
        
    # Draw the ROIs on the frame for visualization
    cv2.rectangle(frame, (signage_roi[0], signage_roi[1]), (signage_roi[2], signage_roi[3]), (128, 0, 128), 2)  # Purple color for signage ROI
    cv2.rectangle(frame, (bus_number_roi[0], bus_number_roi[1]), (bus_number_roi[2], bus_number_roi[3]), (0, 255, 255), 2)  # Yellow color for bus number ROI
    cv2.rectangle(frame, (bus_status_roi[0], bus_status_roi[1]), (bus_status_roi[2], bus_status_roi[3]), (0, 0, 255), 2)  # Red box for bus detection
    for roi in queue_rois:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)  # Green color for queue ROIs

    # Perform inference
    results = model(frame)

    # Parse results
    predictions = results.pred[0]

    # Initialize counters for each queue
    queue_counts = [0] * len(queue_rois)

    # Draw bounding boxes and labels on the frame
    for *box, conf, cls in predictions:
        if cls == 0:  # Assuming class 0 is 'person'
            for i, roi in enumerate(queue_rois):
                if is_inside_roi(box, roi):
                    queue_counts[i] += 1
                    # Draw bounding box for each person in the queue
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    # Display the queue counts on the frame
    for i, count in enumerate(queue_counts):
        label = f'Queue {i+1}: {count}'
        cv2.putText(frame, label, (queue_rois[i][0], queue_rois[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the bus status and number on the frame
    cv2.putText(frame, f'{bus_status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()
