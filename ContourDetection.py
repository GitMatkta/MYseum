import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Function to load reference image from a given file path
def load_reference_image(reference_path):
    return cv2.imread(reference_path)

# Function to load frame images from a specified directory
def load_frame_images(directory_path):
    frame_images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more extensions if necessary
            frame_path = os.path.join(directory_path, filename)
            frame_image = cv2.imread(frame_path)
            if frame_image is not None:
                frame_images.append(frame_image)
    return frame_images

# Function to extract objects from the frame
def extract_objects(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extracted_objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        object_roi = frame[y:y + h, x:x + w]
        extracted_objects.append(object_roi)
    return extracted_objects

# Function to perform histogram matching for each channel (R, G, B)
def match_histograms(image, reference):
    matched_channels = []
    for channel in cv2.split(image):
        # Calculate histograms for the current channel
        matched_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        reference_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(matched_hist, matched_hist, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(reference_hist, reference_hist, 0, 255, cv2.NORM_MINMAX)

        # Perform histogram matching using cv2.calcBackProject
        matched_channel = cv2.calcBackProject([channel], [0], reference_hist, [0, 256], scale=1)
        matched_channel = np.clip(matched_channel, 0, 255)
        matched_channels.append(matched_channel)

    matched_image = cv2.merge(matched_channels)
    return matched_image

# Example usage
reference_image_path = r'C:\University\P3\Project\MYseum\Malerier\Girl_with_a_Pearl_Earring.jpg'  # Provide the path to your reference image
frame_directory_path = r'C:\University\P3\Project\MYseum\IRL'  # Provide the path to the directory containing frame images

# Load reference image
reference_image = load_reference_image(reference_image_path)

# Load frame images from the specified directory
frame_images = load_frame_images(frame_directory_path)

# Process each frame image
for idx, frame in enumerate(frame_images, 1):
    # Extract objects from the frame
    objects = extract_objects(frame)

    # Process each extracted object
    for obj_idx, obj in enumerate(objects, 1):
        # Perform histogram matching for each channel (R, G, B)
        matched_object = match_histograms(obj, reference_image)

        # Calculate histograms for each channel
        hist_b = cv2.calcHist([matched_object], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([matched_object], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([matched_object], [2], None, [256], [0, 256])

        # Plot histograms
        plt.figure(figsize=(8, 6))
        plt.plot(hist_b, color='b', label='Blue Channel')
        plt.plot(hist_g, color='g', label='Green Channel')
        plt.plot(hist_r, color='r', label='Red Channel')
        plt.title(f'Histograms for Object {obj_idx} (Frame {idx})')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # Display the matched object
        plt.imshow(cv2.cvtColor(matched_object, cv2.COLOR_BGR2RGB))
        plt.show()
