import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_channel_percentage(image):
    total_pixels = image.size  # Total number of pixels in the image

    # Calculate the percentage of each color channel
    blue_percentage = (np.sum(image[:, :, 0]) / total_pixels) * 100
    green_percentage = (np.sum(image[:, :, 1]) / total_pixels) * 100
    red_percentage = (np.sum(image[:, :, 2]) / total_pixels) * 100

    return blue_percentage, green_percentage, red_percentage

def plot_histogram(image, title):
    color = ('b', 'g', 'r')
    total_pixels = image.size  # Total number of pixels in the image

    for i, col in enumerate(color):
        # Calculate histogram
        histr = cv.calcHist([image[:, :, i]], [0], None, [256], [0, 256])

        # Normalize histogram values to be percentages
        histr_percentage = (histr / total_pixels) * 100

        # Plot the histogram
        plt.plot(histr_percentage, color=col)
        plt.xlim([0, 256])

    plt.title(title)
    plt.show()

# Folder containing comparison images
comparison_folder = r'malerier'

# Iterate through all images in the comparison folder
for filename in os.listdir(comparison_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the comparison image
        comparison_image = cv.imread(os.path.join(comparison_folder, filename))

        # Calculate and print percentage values for each channel
        blue_percent, green_percent, red_percent = calculate_channel_percentage(comparison_image)
        print(f'Image: {filename}, Blue: {blue_percent:.2f}%, Green: {green_percent:.2f}%, Red: {red_percent:.2f}%')

        # Display histograms for each color channel
        plot_histogram(comparison_image, f'Histograms ({filename})')
