import numpy as np
import os

folder_path = r'some folder'
filename = r'some filename'
file_path = os.path.join(folder_path, filename)
img = cv2.imread(r'SomePic')

# Setting a preset size of the different square regions.
# Can probably be changed or automatically changed with some function :)
image_size = (500, 500)
square_size = image_size[0] // 5

# Dont mind the names, they are used for saving positions
Fields = []
Forest = []
Ocean = []
Unknown = []

# The range of which each painting is defined by the BGR
mona_lisa = [(0, 160, 180), (40, 255, 255)]
girl_with_perl = [(15, 45, 40), (50, 90, 80)]
starry_night = [(120, 70, 0), (255, 145, 30)]

# Go through the image in a 5x5 grid of square regions.
for i in range(5):
    for j in range(5):
        # Calculate the coordinates for the square region.
        y_start = i * square_size
        y_end = (i + 1) * square_size
        x_start = j * square_size
        x_end = (j + 1) * square_size

        # Crop the image to the square region.
        square_region = img[y_start:y_end, x_start:x_end]

        # Reshape the square region to a 1D array of BGR values.
        pixels = square_region.reshape(-1, 3)

        # Calculate the median BGR values within the region.
        median_color = np.median(pixels, axis=0).astype(int)

        # If median color is within the specified ranges, it will count the current region as that landtype.
        mona_lisa_condition = (mona_lisa[0] <= median_color).all() and (median_color <= mona_lisa[1]).all()
        girl_with_perl_condition = (girl_with_perl[0] <= median_color).all() and (median_color <= girl_with_perl[1]).all()
        starry_night_condition = (starry_night[0] <= median_color).all() and (median_color <= starry_night[1]).all()

        # Tells the program what word it should poop out, yeah we hard coded it, deal with it *puts cool sunglasses on*
        if mona_lisa_condition:
            painting = "Mona Lisa"
            Fields.append((i + 1, j + 1))
        elif girl_with_perl_condition:
            painting = "Girl with da Perl"
            Forest.append((i + 1, j + 1))
        elif starry_night_condition:
            painting = "Starry Night"
            Ocean.append((i + 1, j + 1))
        else:
            painting = "Unknown"
            Unknown.append((i + 1, j + 1))

        # another one bites the print.
        print(f"Landscape type for square region {i + 1}-{j + 1} of {filename}: {painting}")
        # This takes the Median color, can be changed with average :)
        print(f"Dominant color for square region {i + 1}-{j + 1} of {filename}: {median_color}")

grid_matrix = [['Unknown' for _ in range(5)] for _ in range(5)]

# Fill in the matrix based on the positions of the landscape types.
for position in Fields:
    i, j = position
    grid_matrix[i - 1][j - 1] = 'Fields'
for position in Forest:
    i, j = position
    grid_matrix[i - 1][j - 1] = 'Forest'
for position in Ocean:
    i, j = position
    grid_matrix[i - 1][j - 1] = 'Ocean'

# Now, grid_matrix contains the landscape types in a 5x5 grid format.
for row in grid_matrix:
    print(row)