import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import csv
import bluetooth
import os
import time

from sklearn.svm import SVC

# Define label_encoder globally
label_encoder = LabelEncoder()

def train_svm_model(data_path):
    # Load the data from CSV
    data = pd.read_csv(data_path, delimiter='\t')

    # Preprocessing
    # Assuming 'Device Name' is the feature and 'Label' is the target variable
    X = data['Device Name']
    y = data['Label']

    # Encoding categorical variables
    global label_encoder  # Add this line
    X_encoded = label_encoder.fit_transform(X)

    # Splitting the dataset into train and test sets
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Support Vector Machine Model
    model = SVC(kernel='linear')

    # Training the model
    model.fit(X_train.reshape(-1, 1), y_train)
    
    return model

def discover_devices():
    devices = bluetooth.discover_devices(
        lookup_names=True, lookup_class=True, device_id=-1)
    return devices


def print_device_info(device_info):
    for addr, name, _ in device_info:
        print(f"Device Name: {name}")
        print("------")
    return [name for _, name, _ in device_info]


def main():
    try:
        while True:
            print("Scanning for Bluetooth devices...")
            devices = discover_devices()
            if devices:
                print(f"Found {len(devices)} devices.")
                device_names = print_device_info(devices)

                # Save device names to a CSV file
                device_names_filename = 'device_names.csv'
                file_exists = os.path.exists(device_names_filename)
                with open(device_names_filename, 'a', newline='') as csvfile:
                    fieldnames = ['Device Name', 'Label']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    model = train_svm_model("device.csv")

                    # Counter for total device counts
                    total_count = 0

                    for name in device_names:
                        try:
                            X = label_encoder.transform([name])  # Corrected this line
                            pred = model.predict(X.reshape(-1, 1))
                            writer.writerow({'Device Name': name, 'Label': pred[0]})

                            # Increment total count if label is Apple, Samsung, or Oppo
                            if pred[0] in ['Apple', 'Samsung', 'Oppo']:
                                total_count += 1

                        except KeyError:
                            # Handle unseen label by classifying as 'Others'
                            writer.writerow({'Device Name': name, 'Label': 'Others'})
                        except ValueError as e:
                            print(f"Error: {e}. Skipping.")

                    print(f"Device names have been {'appended to' if file_exists else 'written to'} '{device_names_filename}'")

                    filename = 'device_count.csv'
                    # Check if the file exists
                    file_exists = os.path.exists(filename)
                    # Write or append the total device count to the CSV file
                    with open(filename, 'a', newline='') as csvfile:
                        fieldnames = ['Total Count']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow({'Total Count': total_count})
                    print(f"Total device count has been {'appended to' if file_exists else 'written to'} '{filename}'")
            else:
                print("No devices found.")

            time.sleep(5)  # Wait for 5 seconds before scanning again

    except KeyboardInterrupt:
        print("Stopping Bluetooth scanner.")


if __name__ == "__main__":
    main()
