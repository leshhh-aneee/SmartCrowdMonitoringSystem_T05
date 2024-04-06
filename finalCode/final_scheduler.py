from yolov5s_withScheduling import BusQueueCounter

def main():
    image_folder_path = '/home/leshane/Downloads/frames'
    device_count_path = 'device_count(latest).csv'
    bus_info_path = 'bus_info.csv'
    wav_file = 'finalVideo.wav'
    model_file = 'bus_detection_model.pkl'
    scaler_file = 'feature_scalar_file.pkl'
    features_csv_file = 'finalVideo_svm_features_final_data_updated.csv'

    bq_counter = BusQueueCounter(image_folder_path, device_count_path, bus_info_path, wav_file, model_file, scaler_file, features_csv_file)
    bq_counter.process_frames((160, 210, 375, 302), (440, 253, 1278, 495), (443, 261, 843, 329), (0, 334, 1294, 693))

if __name__ == "__main__":
    main()
