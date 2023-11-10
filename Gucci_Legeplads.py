import cv2
import numpy as np
import os

irl_path = r"100 Billeder cirka\Almond Blossoms 1.jpg"

reference_folder = r"Malerier"

    # Function for finding the painting in the input image.
def find_painting_in_image(image_path):

    image = cv2.imread(image_path)

        # Check for invalid input image.
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

        # Convert the color image to the HSV (hue, saturation, value) color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper values in HSV format.
    lower_color = np.array([20, 20, 20])
    upper_color = np.array([255, 255, 255])

        # Create a binary mask for thresholding the image.
        # All pixels with a color between lower_color and upper_color become white, all other pixels become black.
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find the contour in the binary mask with the largest area and draw it in blue on the original image.
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest_contour], 0, (255, 0, 0), 2)

        # Extract the largest contour from the original color image by cropping the bounding rectangle.
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_object = image[y:y + h, x:x + w]

        # Calculate the average color of the object and print it in BGR and RGB format.
    average_color = np.mean(cropped_object, axis=(0, 1))
    average_color_rgb = average_color[::-1]
    print(f'Average Color: B={average_color[0]}, G={average_color[1]}, R={average_color[2]}')
    print(f'Average Color (RGB): R={average_color_rgb[0]}, G={average_color_rgb[1]}, B={average_color_rgb[2]}')

        # Return the cropped object.
    return cropped_object

    # Function for splitting the image into a matrix of HSV values.

    # Function to split the image into a matrix of cells with average HSV values.
def split_image_into_matrix(image, rows=10, cols=10):

        # Check for invalid input image.
    if image is None:
        print("Error: Could not open or find the image.")
        return None

        # Resize the image to a size divisible by rows and cols.
    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols
    image = cv2.resize(image, (cell_width * cols, cell_height * rows))

        # Initialize a list for storing the HSV values.
    hsv_values = []

    for i in range(rows):

            # Initialize a list for storing the HSV values in the current row.
        row_values = []

        for j in range(cols):

                # Extract a cell from the image by slicing the image with cell_height and cell_width.
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

                # Convert the color image to the HSV.
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

                # Calculate the average HSV values and add them to the row_values list.
            average_hsv = np.mean(hsv_cell, axis=(0, 1))
            row_values.append(average_hsv)

            # Add the row_values list to the hsv_values list.
        hsv_values.append(row_values)

        # Return the hsv_values list.
    return hsv_values

    # Function for matching the HSV values of the input image with the HSV values of the reference images.
def match_hsv(image, reference_folder):

        # Get the HSV values of the input image using split_image_into_matrix().
    hsv_values1 = split_image_into_matrix(image)

        # Get the reference images from the reference folder using os module.
        # The os module provides functions for interacting with file paths.
        # Iterate over the folder and add the file paths to a list.
    reference_images = [os.path.join(reference_folder, filename) for filename in os.listdir(reference_folder)]

        # Initialize variables for storing the highest similarity percentage and the best match.
    highest_similarity_percentage = 0
    best_match = ""

        # Iterate over the images in the list.
    for reference_image_path in reference_images:

            # Get the HSV values of the reference image using split_image_into_matrix().
        hsv_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if hsv_values1 and hsv_values2:

                # Flatten the matrices for normalized cross-correlation (NCC) calculation.
                # Flattening a matrix means converting a 2D matrix into a 1D array.
                # This is done to make the matrix compatible with the NCC formula.
            flat_hsv_values1 = np.array(hsv_values1).flatten()
            flat_hsv_values2 = np.array(hsv_values2).flatten()

                # Calculate mean for normalization.
                # This is done to make the NCC formula more robust to changes in lighting conditions.
            mean_hsv_values1 = np.mean(flat_hsv_values1)
            mean_hsv_values2 = np.mean(flat_hsv_values2)

                # Calculate Normalized Cross-Correlation (NCC).
            ncc = np.sum((flat_hsv_values1 - mean_hsv_values1) * (flat_hsv_values2 - mean_hsv_values2)) / \
                  (np.sqrt(np.sum((flat_hsv_values1 - mean_hsv_values1) ** 2) * np.sum((flat_hsv_values2 - mean_hsv_values2) ** 2)))

                # Calculate the percentage of similarity of the two images.
            similarity_percentage = (ncc + 1) * 50

            print(f"Similarity between the two images ({reference_image_path}): {similarity_percentage:.2f}%")

                # Update the highest similarity percentage and the best match.
            if similarity_percentage > highest_similarity_percentage:
                highest_similarity_percentage = similarity_percentage
                best_match = os.path.basename(reference_image_path)
        else:
            print(f"Error occurred during image processing for reference image: {reference_image_path}")

        # Print the best match using the filename of the input image and the filename of the best match, and percentage.
    if best_match:
        print(f"///")

        print(
            f"{os.path.basename(irl_path)} has the highest similarity to {best_match} with {highest_similarity_percentage:.2f}%")
    else:
        print(f"Error: No match found for {os.path.basename(image)}")

    # Call find_painting_in_image() on the input image (irl_path).

# ///////////////////////////////////////

image = find_painting_in_image(irl_path)

    # Call match_hsv() on the image (returned from find_painting_in_image()) and the reference folder (reference_folder).
match_hsv(image, reference_folder)

    # Display the image. Close the window by pressing any key.
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
