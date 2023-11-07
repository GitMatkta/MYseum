import cv2
import numpy as np
import os

    # Path to the images
irl_path = r"100 Billeder cirka\Pearl Earring 1.jpg"
reference_folder = r"Malerier"

    # Function to split an image into a matrix of average HSV values
def split_image_into_matrix(image_path, rows=10, cols=10):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}.")
        return None

        # Resize the image to a size divisible by rows and cols.
        # The underscore indicates that we don't care about the third value returned by the shape function, which is the number of color channels.
    height, width, _ = image.shape

        # Calculate the height and width of each cell using the integer division operator //.
        # Integer division is division that results in an integer.
    cell_height = height // rows
    cell_width = width // cols

        # Resize the image to a size divisible by rows and cols.
    image = cv2.resize(image, (cell_width * cols, cell_height * rows))

        # Create an empty list to store the average HSV values.
    hsv_values = []

        # Iterate over the rows and columns of the image.
    for i in range(rows):
        row_values = []
        for j in range(cols):

                # Get the cell at the current row and column.
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

                # Convert the cell to HSV.
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

                # Calculate the average HSV value of the cell.
            average_hsv = np.mean(hsv_cell, axis=(0, 1))

                # Add the average HSV value to the row.
            row_values.append(average_hsv)

            # Add the row to the list of HSV values.
        hsv_values.append(row_values)

    return hsv_values

if __name__ == "__main__":
    # Load the irl_path image and run the function to split the image into matrices of average HSV values
    image_path1 = irl_path
    hsv_values1 = split_image_into_matrix(image_path1)

    # Get a list of image files in the "Malerier" folder
    reference_files = [os.path.join(reference_folder, file) for file in os.listdir(reference_folder) if file.endswith(".jpg")]

    for reference_path in reference_files:
        # Load the reference image and run the function to split the image into matrices of average HSV values
        image_path2 = reference_path
        hsv_values2 = split_image_into_matrix(image_path2)

        # Check if the function returned valid data for both images
        if hsv_values1 and hsv_values2:
            # Flatten the matrices for NCC calculation
            flat_hsv_values1 = np.array(hsv_values1).flatten()
            flat_hsv_values2 = np.array(hsv_values2).flatten()

            # Calculate mean for normalization
            mean_hsv_values1 = np.mean(flat_hsv_values1)
            mean_hsv_values2 = np.mean(flat_hsv_values2)

            # Calculate Normalized Cross-Correlation (NCC)
            ncc = np.sum((flat_hsv_values1 - mean_hsv_values1) * (flat_hsv_values2 - mean_hsv_values2) /
                         (np.sqrt(np.sum((flat_hsv_values1 - mean_hsv_values1) ** 2) * np.sum((flat_hsv_values2 - mean_hsv_values2) ** 2))))

            # Convert NCC to percentage of similarity (ranges from -1 to 1)
            similarity_percentage = (ncc + 1) * 50

            print(f"Similarity between {irl_path} and {reference_path}: {similarity_percentage:.2f}%")
        else:
            print("Error occurred during image processing for either the input or reference image.")

    # Initialize the irl_path image.
image_path = cv2.imread(irl_path)

    # Resize the image with specified width and height.
image = cv2.resize(image_path, (595, 842))

    # Initialize the reference image.
reference_image = cv2.imread(reference_path)

    # Convert the image to grayscale (required for edge detection).
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to the image with the specified thresholds.
edges = cv2.Canny(gray_image, threshold1=340, threshold2=800)

    # Find contours in the edge-detected image.
    # Contours are the boundaries of a shape with the same intensity.
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the frame).
if contours:

        # Find the largest contour (the frame).
    largest_contour = max(contours, key=cv2.contourArea)

        # Draw the largest contour on the image.
    rect = cv2.minAreaRect(largest_contour)

        # Get the angle of the contour
    angle = rect[-1]

        # Get the bounding box of the largest contour.
    x, y, w, h = cv2.boundingRect(largest_contour)

        # Find the four corners of the bounding box and convert them to a numpy array.
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype='float32')

        # Define the target corners for the rectified image (a rectangle).
    target_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')

        # Compute the perspective transformation matrix (homography).
        # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
    transformation_matrix = cv2.getPerspectiveTransform(corners, target_corners)

        # Apply the perspective transformation to rectify the image.
        # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    rectified_image = cv2.warpPerspective(image, transformation_matrix, (w, h))

        # Rotate the rectified image to make it horizontal.
    if angle < -45:
        angle += 90
    rotated_rectified_image = cv2.warpAffine(rectified_image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h))

else:
    print("No contour found")

    # Initialize the threshold for white regions.
white_threshold = 180

    # Threshold the image to remove white regions.
thresholded_image = cv2.inRange(rotated_rectified_image, (white_threshold, white_threshold, white_threshold), (255, 255, 255))

    # Invert the thresholded image (so white regions become black and vice versa).
thresholded_image = cv2.bitwise_not(thresholded_image)

    # Apply the thresholded mask to the rotated rectified image.
removed_white = cv2.bitwise_and(rotated_rectified_image, rotated_rectified_image, mask=thresholded_image)

    # Convert the "removed_white" image to grayscale.
gray_removed_white = cv2.cvtColor(removed_white, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale image.
contours, _ = cv2.findContours(gray_removed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour.
if contours:
        # Find the largest contour (the frame).
    largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour.
    x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the painting from the "removed_white" image
    cropped_painting = removed_white[y:y+h, x:x+w]

        # Display the cropped painting
    cv2.imshow("Cropped Painting", cropped_painting)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contour found")