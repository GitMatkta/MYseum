import cv2


# Function to find and outline rectangular objects in an image
def find_and_outline_object(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a rectangle detection algorithm (such as contour detection) to find the rectangular object
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find the rectangular object
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # If the object has four vertices, it's likely a rectangle
        if len(approx) == 4:
            # Draw a green outline around the rectangular object
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

            # Extract and process the rectangular object
            x, y, w, h = cv2.boundingRect(approx)
            object_roi = image[y:y + h, x:x + w]

            # Implement your logic to analyze the object_roi here
            # For example, you can perform further image processing or use a machine learning model

            # Display the outlined object
            cv2.imshow("Outlined Object", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Path to the input image
image_path = r"C:\University\P3\Project\MYseum\IRL\Girl with da perl training data.jpg"

# Call the function to find and outline the object in the image
find_and_outline_object(image_path)
