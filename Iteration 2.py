import cv2
import numpy as np
import os

# Improved contour detection by thresholding away the white background and identifying the white pixels.
# Then finding the coordinates of the furthest white pixels and drawing a rectangle around them.
# This is done by finding the minimum and maximum coordinates of the white pixels.
# Changed HSV matching to RGB matching.

# Restraints:
# All input images in the input folder must be the same size.
# All input images must be taken by the same mobile device.
# The input images must be taken in the same lighting conditions.
# The input images must be taken at the same distance from the painting.
# The input images must be taken at a maximum angle of approximately 45 degrees from the painting.

    # Initialize the paths to the input folder and the reference folder.
input_folder = r"100 Billeder cirka"
reference_folder = r"Malerier"


def find_painting_in_image(image_path):

        # Initialize the image variable using cv2.imread().
    image = cv2.imread(image_path)

        # Use cvtColor() to convert the image into HSV (hue, saturation, value) color space for easier color detection.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper HSV values.
        # The lower_color HSV values are 50 to threshold for the white background (restraint) in the image.
    lower_color = np.array([50, 50, 50])
    upper_color = np.array([255, 255, 255])

        # Create a binary mask for thresholding the image using the previously defined lower and upper HSV values.
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Use where() to find the coordinates of all white pixels (>0).
        # Use column_stack() to stack the coordinates in a 2D array.
        # Store the coordinates in a numpy array called white_pixel_coordinates.
    white_pixel_coordinates = np.column_stack(np.where(color_mask > 0))

        # Define the bottom left and top right coordinates of the painting in the image using min() and max().
    bottom_left = np.min(white_pixel_coordinates, axis=0)
    top_right = np.max(white_pixel_coordinates, axis=0)

        # Define a region of interest (roi) using the previously defined coordinates.
    roi = image[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]

        # Draw a green rectangle around the region of interest.
        # The rectangle is drawn using cv2.rectangle() and the coordinates are reversed using [::-1].
        # The coordinates are reversed because cv2.rectangle() takes the coordinates in the format (x, y),
        # while the coordinates are stored in the format (y, x) in the numpy array.
    cv2.rectangle(image, tuple(bottom_left[::-1]), tuple(top_right[::-1]), (0, 255, 0), 2)

        # Return the region of interest (roi).
    return roi
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

            # Initialize a list for storing the RGB values.
        rgb_values = []

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

                    # Use mean() to calculate the average RGB values and add them to the row_values list.
                average_rgb = np.mean(cell, axis=(0, 1))

                    # Append the average RGB values to the row_values list.
                row_values.append(average_rgb)

                # Append the row_values list to the rgb_values list.
            rgb_values.append(row_values)

            # Return the rgb_values list.
        return rgb_values
def match_rgb(image, reference_folder):
    global imageNumber

    # Get the RGB values of the input image using split_image_into_matrix().
    rgb_values1 = split_image_into_matrix(image)

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
        rgb_values2 = split_image_into_matrix(cv2.imread(reference_image_path))

        if rgb_values1 and rgb_values2:

                # Flatten the matrices for normalized cross-correlation (NCC) calculation.
                # Flattening a matrix means converting a 2D matrix into a 1D array.
                # This is done to make the matrix compatible with the NCC formula.
                # NCC documentation: https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
            flat_rgb_values1 = np.array(rgb_values1).flatten()
            flat_rgb_values2 = np.array(rgb_values2).flatten()

                # Perform NCC calculation using the formula from the documentation.
            ncc = np.sum(flat_rgb_values1 * flat_rgb_values2) / (
                    np.sqrt(np.sum(flat_rgb_values1 ** 2) * np.sum(flat_rgb_values2 ** 2)))

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
        text = f"Best Match: {best_match} ({highest_similarity_percentage:.2f}%)"

            # Draw the text on the image using cv2.putText().
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"///")
        print(
            f"\"{os.path.basename(image_path)}\" has the highest similarity to \"{best_match}\" with {highest_similarity_percentage:.2f}%")
        print(f"///")


        if best_match == os.path.basename(r"Malerier/Almond Blossoms.jpg"):
            imageNumber = 0
        if best_match == os.path.basename(r"Malerier/Cafe Terrace at Night.jpg"):
            imageNumber = 1
        if best_match == os.path.basename(r"Malerier/Girl_with_a_Pearl_Earring.jpg"):
            imageNumber = 2
        if best_match == os.path.basename(r"Malerier/Impression, Sunrise - Claude Monet.jpg"):
            imageNumber = 3
        if best_match == os.path.basename(r"Malerier/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg.jpg"):
            imageNumber = 4
        if best_match == os.path.basename(r"Malerier/Monet Bridge over pond of water lilies.jpg"):
            imageNumber = 5
        if best_match == os.path.basename(r"Malerier/Starry Night - Van_Gogh.jpg"):
            imageNumber = 6
        if best_match == os.path.basename(r"Malerier/Tower of Babel.jpg"):
            imageNumber = 7
        if best_match == os.path.basename(r"Malerier/Wanderer_above_the_sea_of_fog - Caspar_David_Friedrich_.jpg"):
            imageNumber = 8
        if best_match == os.path.basename(r"Malerier/Almond Blossoms.jpg"):
            imageNumber = 9


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
    match_rgb(image, reference_folder)

        # Show the image in a window, and wait for a keypress, then close the window.
    cv2.imshow("Image", image)
    print(imageNumber)
    cv2.waitKey(0)
cv2.destroyAllWindows()
