# preprocessing.py

import os
import cv2

def extract_frames(input_folder, output_folder, resize_shape=(224, 224)):
    for folder in os.listdir(input_folder):
        input_path = os.path.join(input_folder, folder)
        output_path = os.path.join(output_folder, folder)
        os.makedirs(output_path, exist_ok=True)
        for video_file in os.listdir(input_path):
            video_path = os.path.join(input_path, video_file)
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 0
            while success:
                resized_image = cv2.resize(image, resize_shape)
                frame_path = os.path.join(output_path, f"{video_file}_frame{count}.jpg")
                cv2.imwrite(frame_path, resized_image)
                success, image = vidcap.read()
                count += 1
            vidcap.release()


extract_frames("data/Celeb-real", "data/Celeb-real_frames")
extract_frames("data/Celeb-synthesis", "data/Celeb-synthesis_frames")
