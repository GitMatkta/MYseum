import cv2
import numpy as np
import os

irl_path = r"100 Billeder cirka\Almond Blossoms 8.jpg"

reference_folder = r"Malerier"
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
def match_hsv(image, reference_folder):
    hsv_values1 = split_image_into_matrix(image)

    reference_images = [os.path.join(reference_folder, filename) for filename in os.listdir(reference_folder)]

    for reference_image_path in reference_images:

        hsv_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if hsv_values1 and hsv_values2:

            flat_hsv_values1 = np.array(hsv_values1).flatten()

            flat_hsv_values2 = np.array(hsv_values2).flatten()

            mean_hsv_values1 = np.mean(flat_hsv_values1)
            mean_hsv_values2 = np.mean(flat_hsv_values2)

            ncc = np.sum((flat_hsv_values1 - mean_hsv_values1) * (flat_hsv_values2 - mean_hsv_values2)) / \
                  (np.sqrt(np.sum((flat_hsv_values1 - mean_hsv_values1) ** 2) * np.sum((flat_hsv_values2 - mean_hsv_values2) ** 2)))

            similarity_percentage = (ncc + 1) * 50

            print(f"Similarity between the two images ({reference_image_path}): {similarity_percentage:.2f}%")
        else:

            print(f"Error occurred during image processing for reference image: {reference_image_path}")
def find_painting_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([20, 20, 20])

    upper_color = np.array([255, 255, 255])

    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(image, [largest_contour], 0, (255, 0, 0), 2)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_object = image[y:y + h, x:x + w]

    average_color = np.mean(cropped_object, axis=(0, 1))

    average_color_rgb = average_color[::-1]

    print(f'Average Color: B={average_color[0]}, G={average_color[1]}, R={average_color[2]}')

    print(f'Average Color (RGB): R={average_color_rgb[0]}, G={average_color_rgb[1]}, B={average_color_rgb[2]}')

    return cropped_object

image = find_painting_in_image(irl_path)
match_hsv(image, reference_folder)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
