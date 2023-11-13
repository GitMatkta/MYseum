import cv2
import numpy as np
import os

irl_folder = r"100 Billeder cirka"

reference_folder = r"Malerier"

def find_painting_in_image(image_path):

        # Load the image using cv2.imread().
    image = cv2.imread(image_path)

        # Check for invalid input image (failsafe for cv2.imread()
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

        # Convert the image variable into HSV (hue, saturation, value) color spave.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper HSV values. lower_color HSV values are 50 to threshold for the white in the image.
    lower_color = np.array([50, 50, 50])
    upper_color = np.array([255, 255, 255])

        # Create a binary mask for thresholding the image using the previously defined lower and upper HSV values.
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Finds the coordinates of all white pixels (>0) in the binary mask and adds them to a list (numpy array).
        # This uses np.column_stack() to stack the coordinates in a 2D array and np.where() to find the coordinates.
    white_pixel_coordinates = np.column_stack(np.where(color_mask > 0))

        # Define the bottom left and top right coordinates of the painting in the image using np.min() and np.max().
    bottom_left = np.min(white_pixel_coordinates, axis=0)
    top_right = np.max(white_pixel_coordinates, axis=0)

        # Create a region of interest in the image using the previously defined coordinates.
    roi = image[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]

        # Draw a green rectangle around the region of interest.
        # The rectangle is drawn using cv2.rectangle() and the coordinates are reversed using [::-1].
    cv2.rectangle(image, tuple(bottom_left[::-1]), tuple(top_right[::-1]), (0, 255, 0), 2)

        # Return the region of interest.
    return roi

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

def match_rgb(image, reference_folder):
    # Get the RGB values of the input image using split_image_into_matrix().
    rgb_values1 = split_image_into_matrix(image)

    # Get the reference images from the reference folder using os module.
    reference_images = [os.path.join(reference_folder, filename) for filename in os.listdir(reference_folder)]

    # Initialize variables for storing the highest similarity percentage and the best match.
    highest_similarity_percentage = 0
    best_match = ""

    # Iterate over the images in the list.
    for reference_image_path in reference_images:
        # Get the RGB values of the reference image using split_image_into_matrix().
        rgb_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if rgb_values1 and rgb_values2:
            # Flatten the matrices for normalized cross-correlation (NCC) calculation.
            flat_rgb_values1 = np.array(rgb_values1).flatten()
            flat_rgb_values2 = np.array(rgb_values2).flatten()

            # Calculate Normalized Cross-Correlation (NCC).
            ncc = np.sum(flat_rgb_values1 * flat_rgb_values2) / (
                    np.sqrt(np.sum(flat_rgb_values1 ** 2) * np.sum(flat_rgb_values2 ** 2)))

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
        text = f"Best Match: {best_match} ({highest_similarity_percentage:.2f}%)"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"///")
        print(
            f"\"{os.path.basename(image_path)}\" has the highest similarity to \"{best_match}\" with {highest_similarity_percentage:.2f}%")
        print(f"///")

    else:
        print(f"Error: No match found for {os.path.basename(image)}")

# Get a list of all image files in the input folder
image_files = [os.path.join(irl_folder, filename) for filename in os.listdir(irl_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# Iterate over each image in the folder
for image_path in image_files:
    # Find the painting in the current image
    image = find_painting_in_image(image_path)

    # Match the RGB values and find the best match
    match_rgb(image, reference_folder)

    # Show the image in a window
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Close all windows when the processing is done
cv2.destroyAllWindows()
