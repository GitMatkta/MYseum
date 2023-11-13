import cv2
import numpy as np

irl_path = r"100 Billeder cirka\Wanderer 8.jpg"

reference_folder = r"Malerier"

    # Function for finding the painting in the input image.
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

    # Call the find_painting_in_image() function on the input image and save the result in the image variable.
image = find_painting_in_image(irl_path)

    # Show the image in a window.
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
