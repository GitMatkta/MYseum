import cv2
import numpy as np
import os


# Function to extract SIFT features from an image
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


# Function to match features between the input image and a database image
def match_features(input_descriptor, db_descriptor):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(input_descriptor, db_descriptor, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


# Path to the input image
input_image_path = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\IRL\Girl with da perl training data.jpg"

# Path to the folder containing database paintings
database_folder = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\Malerier"

# Check if the input image file exists
if not os.path.isfile(input_image_path):
    print("Error: The input image file does not exist.")
else:
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Check if the input image was loaded successfully
    if input_image is not None:
        input_keypoints, input_descriptors = extract_sift_features(input_image)

        # Initialize variables to keep track of the best match
        best_match = None
        best_match_score = 0

        # Loop through the database folder
        for filename in os.listdir(database_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                db_image = cv2.imread(os.path.join(database_folder, filename))

                # Check if the database image was loaded successfully
                if db_image is not None:
                    db_keypoints, db_descriptors = extract_sift_features(db_image)

                    # Match features and get the number of good matches
                    match_score = match_features(input_descriptors, db_descriptors)

                    # Update the best match if the current match is better
                    if match_score > best_match_score:
                        best_match = filename
                        best_match_score = match_score

        # Set a threshold for a valid match (you can adjust this)
        threshold = 100

        # Check if the best match score exceeds the threshold
        if best_match_score > threshold:
            print("Best match:", best_match)
        else:
            print("No matching painting found in the database.")
    else:
        print("Error: The input image could not be loaded.")
