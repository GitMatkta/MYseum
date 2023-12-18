import cv2
import numpy as np
import os

    # Initialize the paths to the input folder and the reference folder.
input_folder = r"100 Billeder cirka"
reference_folder = r"Malerier"

def find_painting_in_image(image_path):

        # Initialize the image variable using cv2.imread().
    image = cv2.imread(image_path)

        # Use cvtColor() to convert the image into HSV (hue, saturation, value) color space for easier color detection.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper HSV values.
        # The lower_color HSV values are 20 to threshold for the white background (restraint) in the image.
    lower_color = np.array([20, 20, 20])
    upper_color = np.array([255, 255, 255])

        # Create a binary mask for thresholding the image using the previously defined lower and upper HSV values.
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Use findContours() to find the contours in the image using the binary mask.
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find, draw, and crop the largest contour in the image.
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest_contour], 0, (255, 0, 0), 2)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_object = image[y:y + h, x:x + w]

        # Return the cropped object.
    return cropped_object
def split_image_into_matrix(image, rows=10, cols=10):

        # Define a height and width variable for the image using shape().
    height, width, _ = image.shape

        # Define the height and width of each cell in the image.
        # Calculate cell size by dividing height and width by the number of rows and columns.
    cell_height = height // rows
    cell_width = width // cols

        # Resize the image to a size divisible by rows and cols using resize().
        # Resizing is done in order to make the image easier to split into a grid.
    image = cv2.resize(image, (cell_width * cols, cell_height * rows))

        # Initialize a list for storing the HSV values.
    hsv_values = []

        # Iterate through the rows and columns in the image.
    for i in range(rows):

            # Initialize a list for storing the RGB values in the current row.
        row_values = []

            # Iterate through the columns in the current row.
        for j in range(cols):

                # Extract a cell from the image by slicing the image with cell_height and cell_width.
                # This uses the slicing syntax [start:stop].
                # The syntax includes the start value and excludes the stop value.
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

                # Use cvtColor() to convert the cell into HSV.
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

                # Calculate the average HSV values for the cell using mean().
            average_hsv = np.mean(hsv_cell, axis=(0, 1))

                # Append the average HSV values to the row_values list.
            row_values.append(average_hsv)

            # Append the row_values list to the rgb_values list.
        hsv_values.append(row_values)

        # Return the rgb_values list.
    return hsv_values
def match_hsv(image, reference_folder):

        # Get the HSV values of the input image using split_image_into_matrix().
    hsv_values1 = split_image_into_matrix(image)

        # Use the os module to get the reference images from the reference folder.
        # This is done by iterating through the files in the reference folder using os.listdir()
        # and using os.path.join() to join the folder path with the filename.
    reference_images = [os.path.join(reference_folder, filename) for filename in os.listdir(reference_folder)]

        # Initialize variables for storing the highest similarity percentage and the best match filename.
    highest_similarity_percentage = 0
    best_match = ""

        # Iterate over the images in the reference_images list.
    for reference_image_path in reference_images:

            # Get the RGB values of the reference image using split_image_into_matrix().
        hsv_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if hsv_values1 and hsv_values2:

                # Flatten the matrices for normalized cross-correlation (NCC) calculation.
                # Flattening a matrix means converting a 2D matrix into a 1D array.
                # This is done to make the matrix compatible with the NCC formula.
                # NCC documentation: https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
            flat_hsv_values1 = np.array(hsv_values1).flatten()
            flat_hsv_values2 = np.array(hsv_values2).flatten()

                # Calculate the mean HSV values for the input image and the reference image for NCC calculation.
            mean_hsv_values1 = np.mean(flat_hsv_values1)
            mean_hsv_values2 = np.mean(flat_hsv_values2)

                # Perform NCC calculation using the formula from the documentation.
                # The result of the NCC calculation is typically between -1 and 1.
                # By adding 1 and multiplying by 50, the result is converted to a percentage between 0 and 100.
            ncc = np.sum((flat_hsv_values1 - mean_hsv_values1) * (flat_hsv_values2 - mean_hsv_values2)) / \
                  (np.sqrt(np.sum((flat_hsv_values1 - mean_hsv_values1) ** 2) * np.sum((flat_hsv_values2 - mean_hsv_values2) ** 2)))

                # Calculate the percentage of similarity of the two images.
            similarity_percentage = (ncc + 1) * 50

                # Print the similarity percentage between the two images.
            print(f"Similarity between the two images ({reference_image_path}): {similarity_percentage:.2f}%")

                # If the similarity percentage is higher than the highest_similarity_percentage,
            if similarity_percentage > highest_similarity_percentage:

                    # Update the highest_similarity_percentage and the best match.
                highest_similarity_percentage = similarity_percentage
                best_match = os.path.basename(reference_image_path)
        else:
                # Print an error message if an error occurs during image processing.
            print(f"Error occurred during image processing for reference image: {reference_image_path}")

    if best_match:

            # If a match is found, print the best match using the filename of the input image and the filename of the best match, and percentage.
        print(f"///")
        print(
            f"\"{os.path.basename(image_path)}\" has the highest similarity to \"{best_match}\" with {highest_similarity_percentage:.2f}%")
    else:
            # Print an error message if no match is found (should be impossible though).
        print(f"Error: No match found for {os.path.basename(image)}")

    # Use the os module to get the image files from the input folder.
image_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# Iterate over the image files.
for image_path in image_files:

            # Find the painting in the image using find_painting_in_image().
        image = find_painting_in_image(image_path)

            # Match the painting in the image to the reference images using match_rgb().
        match_hsv(image, reference_folder)

                # Show the image in a window, and wait for a keypress, then close the window.
        cv2.imshow("Image", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()
