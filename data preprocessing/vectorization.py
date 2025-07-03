import cv2 as cv
import numpy as np
import pandas as pd
import os


resize_shape = (64, 64)
output_csv = "combined_vectors.csv"

image_folder = r"source/images"
video_folder = r"source/videos"

data_rows = []
columns = ["source", "type"] + [f"pixel_{i}" for i in range(resize_shape[0] * resize_shape[1])]


for file in os.listdir(image_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, file)
        img = cv.imread(image_path)
        if img is None:
            print(f" Warning: Couldn't read {file}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, resize_shape)
        vector = resized.flatten().tolist()
        data_rows.append([file, "image"] + vector)


for video_file in os.listdir(video_folder):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_file}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            resized = cv.resize(gray, resize_shape)
            vector = resized.flatten().tolist()
            frame_label = f"{video_file}_frame_{frame_count}"
            data_rows.append([frame_label, "video"] + vector)
            frame_count += 1
        cap.release()
        print(f" Processed {frame_count} frames from {video_file}")


df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_csv, index=False)

