import cv2
import numpy as np

def split_image_into_matrix(image_path, rows=10, cols=10):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}.")
        return None

    # Resize the image to a size divisible by rows and cols
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

if __name__ == "__main__":
    # First image
    image_path1 = "100 Billeder cirka\Pearl Earring 1.jpg"
    hsv_values1 = split_image_into_matrix(image_path1)

    # Second image
    image_path2 = "Malerier\Girl_with_a_Pearl_Earring.jpg"
    hsv_values2 = split_image_into_matrix(image_path2)

    if hsv_values1 and hsv_values2:
        # Flatten the matrices for NCC calculation
        flat_hsv_values1 = np.array(hsv_values1).flatten()
        flat_hsv_values2 = np.array(hsv_values2).flatten()

        # Calculate mean for normalization
        mean_hsv_values1 = np.mean(flat_hsv_values1)
        mean_hsv_values2 = np.mean(flat_hsv_values2)

        # Calculate Normalized Cross-Correlation
        ncc = np.sum((flat_hsv_values1 - mean_hsv_values1) * (flat_hsv_values2 - mean_hsv_values2)) / \
              (np.sqrt(np.sum((flat_hsv_values1 - mean_hsv_values1) ** 2) * np.sum((flat_hsv_values2 - mean_hsv_values2) ** 2)))

        similarity_percentage = (ncc + 1) * 50  # Convert NCC to percentage of similarity (ranges from -1 to 1)

        print("Similarity between the two images: {:.2f}%".format(similarity_percentage))
    else:
        print("Error occurred during image processing.")
