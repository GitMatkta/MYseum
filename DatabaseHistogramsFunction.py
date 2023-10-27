import cv2
import os

# Function to extract BGR histograms from images in a folder
def extract_histograms_from_folder(folder_path):

    # Create a list to store the BGR histograms and file names
    bgr_histograms = []
    file_names = []

    # Loop through all the files in the folder using os.listdir
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):

            # Using os.path.join to create the full path by concatenating the folder path and the filename.
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Check if image is not None
            if image is not None:
                # Calculate the BGR histogram.
                histogram = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                    # [image] is the image,
                    # [0, 1, 2] is the channels/colors to be used,
                    # None is the mask,
                    # [256, 256, 256] is the size of the histogram,
                    # [0, 256, 0, 256, 0, 256] is the range of the histogram

                # Append the histogram to the list
                bgr_histograms.append(histogram)

                file_names.append(filename)

    return bgr_histograms