import cv2
import numpy as np

irl_path = r"100 Billeder cirka\Almond Blossoms 8.jpg"
reference_path = r"Malerier\Almond Blossoms.jpg"

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
    image_path1 = irl_path
    hsv_values1 = split_image_into_matrix(image_path1)

    # Second image
    image_path2 = reference_path
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


# Section 1: Load and Preprocess the Image

# Load an image from your file system
image_path = cv2.imread(irl_path)
image = cv2.resize(image_path, (595, 842))

# Load a reference image
reference_image = cv2.imread(reference_path)

# Convert the image to grayscale (required for edge detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Section 2: Edge Detection

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(gray_image, threshold1=340, threshold2=800)  # You can adjust the thresholds

# Section 3: Contour Detection

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the frame)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Section 4: Rectification

    # Get the orientation angle of the largest contour
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Find the four corners of the bounding box
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype='float32')

    # Define the target corners for the rectified image (a rectangle)
    target_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')

    # Compute the perspective transformation matrix (homography)
    M = cv2.getPerspectiveTransform(corners, target_corners)

    # Apply the perspective transformation to rectify the image
    rectified_image = cv2.warpPerspective(image, M, (w, h))

    # Rotate the rectified image to make it horizontal
    if angle < -45:
        angle += 90
    rotated_rectified_image = cv2.warpAffine(rectified_image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h))

else:
    print("No contour found")

# Section 5: White Region Removal

# Define the upper threshold for white color
white_threshold = 180

# Threshold the image to remove white regions
thresholded_image = cv2.inRange(rotated_rectified_image, (white_threshold, white_threshold, white_threshold), (255, 255, 255))

# Invert the thresholded image (so white regions become black and vice versa)
thresholded_image = cv2.bitwise_not(thresholded_image)

# Apply the thresholded mask to your original image
removed_white = cv2.bitwise_and(rotated_rectified_image, rotated_rectified_image, mask=thresholded_image)

# Section 6: Painting Cropping

# Convert the "removed_white" image to grayscale
gray_removed_white = cv2.cvtColor(removed_white, cv2.COLOR_BGR2GRAY)

# Find contours in the grayscale image
contours, _ = cv2.findContours(gray_removed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the painting)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
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