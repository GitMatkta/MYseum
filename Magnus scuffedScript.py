import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

irl_folder = r"100 Billeder cirka"
reference_folder = r"Malerier"

image_files = [os.path.join(irl_folder, filename) for filename in os.listdir(irl_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

def find_painting_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([50, 50, 50])
    upper_color = np.array([255, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    white_pixel_coordinates = np.column_stack(np.where(color_mask > 0))
    bottom_left = np.min(white_pixel_coordinates, axis=0)
    top_right = np.max(white_pixel_coordinates, axis=0)
    roi = image[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]
    cv2.rectangle(image, tuple(bottom_left[::-1]), tuple(top_right[::-1]), (0, 255, 0), 2)
    return roi

def split_image_into_matrix(image, rows=10, cols=10):
    if image is None:
        print("Error: Could not open or find the image.")
        return None

    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols
    image = cv2.resize(image, (cell_width * cols, cell_height * rows))
    hsv_values = []

    for i in range(rows):
        row_values = []
        for j in range(cols):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            average_hsv = np.mean(hsv_cell, axis=(0, 1))
            row_values.append(average_hsv)
        hsv_values.append(row_values)

    return hsv_values

def match_rgb(image, reference_folder):
    rgb_values1 = split_image_into_matrix(image)
    reference_images = [os.path.join(reference_folder, filename) for filename in os.listdir(reference_folder)]
    highest_similarity_percentage = 0
    best_match = ""

    for reference_image_path in reference_images:
        rgb_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if rgb_values1 and rgb_values2:
            flat_rgb_values1 = np.array(rgb_values1).flatten()
            flat_rgb_values2 = np.array(rgb_values2).flatten()
            ncc = np.sum(flat_rgb_values1 * flat_rgb_values2) / (
                    np.sqrt(np.sum(flat_rgb_values1 ** 2) * np.sum(flat_rgb_values2 ** 2)))
            similarity_percentage = (ncc + 1) * 50

            print(f"Similarity between the two images ({reference_image_path}): {similarity_percentage:.2f}%")

            if similarity_percentage > highest_similarity_percentage:
                highest_similarity_percentage = similarity_percentage
                best_match = os.path.basename(reference_image_path)
        else:
            print(f"Error occurred during image processing for reference image: {reference_image_path}")

    if best_match:
        text = f"Best Match: {best_match} ({highest_similarity_percentage:.2f}%)"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"///")
        print(
            f"\"{os.path.basename(image_path)}\" has the highest similarity to \"{best_match}\" with {highest_similarity_percentage:.2f}%")
        print(f"///")
    else:
        print(f"Error: No match found for {os.path.basename(image)}")

def process_image(image_path):
    image = find_painting_in_image(image_path)
    if image is not None:
        match_rgb(image, reference_folder)
        return image
    else:
        print(f"Error processing image at {image_path}: No painting found.")
        return None

# Batch processing
batch_size = 10  # Adjust this value based on your system resources

processed_images = []  # Initialize an empty list to store processed images

for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i + batch_size]

    with ThreadPoolExecutor() as executor:
        processed_batch = list(executor.map(process_image, batch_files))

    processed_images.extend(processed_batch)  # Append the processed batch to the list

# Iterate over each processed image
for processed_image in processed_images:
    # No need to match RGB values here, as it has already been done in process_image function
    pass

# Close all windows when the processing is done
cv2.destroyAllWindows()
