# Dimitrios Ioannidis
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import os 
from scipy.ndimage import maximum_filter

from deliverable_1 import my_hough_transform
from deliverable_2 import my_corner_harris, my_corner_peaks
from deliverable_3 import rot_img


# Checking if four points form a rectangle (or close to a rectangle)
def is_rectangle(pts, img):

    if len(pts) != 4:
        return False

    # Computing the centroid of the points
    centroid = pts.mean(axis=0)

    # Computing the angles of the points
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

    # Sorting the points by their angles
    pts = pts[np.argsort(angles)]

    # Computing the vectors of the sides
    vectors = pts - np.roll(pts, -1, axis=0)

    # Checking whether consecutive vectors are perpendicular (using dot product)
    for i in range(4):
        dotprod = np.dot(vectors[i], vectors[(i + 1) % 4]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i + 1) % 4]))
        if abs(dotprod) > 0.02:
            return False
        
    # Computing the distances between pairs of points
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    dists = np.array(dists)

    # Sorting the distances
    dists.sort()

    # Defining a difference tolerance for the distances
    tol = np.sqrt(img.shape[0]**2 + img.shape[1]**2) * 0.02

    # For a rectangle, we should have two pairs of equal sides and equal diagonals
    if abs(dists[0]-dists[1]) <= tol and abs(dists[2]-dists[3]) <= tol and (dists[4]-dists[5]) <= tol:
        return True

    return False

# Finding rectangles formed by the harris corners and which of these rectangles contain a picture
def find_pictures(corners, img):

    picture_rectangles = []
    rectangles = []
    for pts in combinations(corners, 4):
        pts = np.array(pts)
        
        # Compute the centroid of the points
        centroid = pts.mean(axis=0)

        # Compute the angles of the points
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

        # Sort the points by their angles
        pts = pts[np.argsort(angles)]
        
        if is_rectangle(pts, img):
            rectangles.append(pts)

    # Filtering out rectangles that don't contain an image or that contain more than one image
    # We first apply a binary threshold to all rectangle areas and compute the ratio of white pixels to the total area, 
    # and filter out images that have more a ratio larger than 0.1
    # We then check which rectangles have shared corners. For rectangles with shared corners, 
    # keep the one with the fewest white pixels

    picture_data = [] # List of tuples (rectangle, white_pixel_ratio)

    center = np.array([img.shape[0] / 2, img.shape[1] / 2]) # Center of the image

    for rectangle in rectangles:
        x_min = int(min(rectangle[:, 1]))
        x_max = int(max(rectangle[:, 1]))
        y_min = int(min(rectangle[:, 0]))
        y_max = int(max(rectangle[:, 0]))

        # Computing the area of the rectangle
        area = (x_max - x_min) * (y_max - y_min)

        # Computing the angle of rotation
        dy = rectangle[1][0] - rectangle[0][0]
        dx = rectangle[1][1] - rectangle[0][1]
        angle = -np.arctan2(dx, dy)

        rotated_image = rot_img(img, angle)

        rotational_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Computing the center of the image
        center_rot = np.array([rotated_image.shape[0] / 2, rotated_image.shape[1] / 2]) # Center of the rotated image

        # Computing the rotated rectangle
        rotated_rectangle = rotational_matrix @ (rectangle - center).T + center_rot.reshape(-1, 1)

        # Computing the bounding box
        x_min = int(min(rotated_rectangle[1, :]))
        x_max = int(max(rotated_rectangle[1, :]))
        y_min = int(min(rotated_rectangle[0, :]))
        y_max = int(max(rotated_rectangle[0, :]))

        # Cropping the picture
        picture = rotated_image[y_min:y_max, x_min:x_max]
        
        # Applying a binary threshold to the picture
        img_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY)

        # Computing the ratio of white pixels to the total area
        white_pixels = np.sum(img_binary == 255)
        white_pixel_ratio = white_pixels / area

        # If more than 20% of the pixels are white, we filter it out
        if white_pixel_ratio <= 0.2:
            picture_data.append((rectangle, white_pixel_ratio, picture))
            

    # Sorting pictures by the number of white pixels
    picture_data.sort(key=lambda x: x[1])

    # Filtering out pictures with shared corners
    # The remaining pictures are tuples in the form (rectangle, white_pixel_ratio, picture), saved in the final_picture_data list
    final_picture_data = []
    
    for i, picture_data_i in enumerate(picture_data):

        rectangle_i, white_pixel_ratio_i, _ = picture_data_i

        corner_matches_found = False # Flag for whether two pictures share more than 1 corner
        for j, picture_data_j in enumerate(final_picture_data):
            rectangle_j, white_pixel_ratio_j, _ = picture_data_j

            # Finding the number of shared corners between picture_i and picture_j
            shared_corners = sum(np.array_equal(corner_i, corner_j) for corner_i in rectangle_i for corner_j in rectangle_j)

            # If more than 1 corner is shared, compare the white pixel ratios and keep the picture with the smallest one.
            if shared_corners > 1: 
                if white_pixel_ratio_i < white_pixel_ratio_j:
                    final_picture_data.remove(picture_data_j)
                    final_picture_data.append(picture_data_i)
                corner_matches_found = True
                break
        
        # If the picture doesn't share more than 1 corner with another picture, add it to the final list
        if not corner_matches_found:
            final_picture_data.append(picture_data_i)

    # Extracting the rectangles and pictures from the final_picture_data list and return them
    picture_rectangles = [rectangle for rectangle, _, _ in final_picture_data]

    pictures = [picture for _, _, picture in final_picture_data]

    return pictures, picture_rectangles


def my_lazy_scanner(imgpath):
    print(f"\nPROCESSING {imgpath}..")
    base_name = os.path.splitext(os.path.basename(imgpath))[0]  # Extracting the base name of the input image file

    # Loading the image and converting to RGB
    img_read = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img_original = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

    # Scaling image down
    scale_percent = 30 # percent of original size
    width = int(img_original.shape[1] * scale_percent / 100)
    height = int(img_original.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_original = cv2.resize(img_original, dim, interpolation = cv2.INTER_AREA)

    ## HOUGH TRANSFORM
    # Converting the image to grayscale
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Applying a binary threshold to the image
    # Pixels with a value greater than 235 are set to 255, all other pixels are set to 0
    _, img_binary = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY)

    # Applying a Gaussian filter to the binary image to get rid of noise
    img_binary_filtered = cv2.GaussianBlur(img_binary, (27,27), 0)

    # Applying the Canny edge detector
    edges = cv2.Canny(img_binary_filtered, 100, 200)
    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title(f"Edge Detection Result for {imgpath}")
    # plt.imsave(f'Report/Images/{base_name}_edges.png', edges, cmap='gray')

    # Variable initialization for the case of overlapping pictures, where Hough Transform will be used
    lines = np.zeros(np.shape(edges))
    hough_transform_applied = False

    ## Finding Harris Corners with our custom implementation
    # The input image for the harris corner detector is the "edges" image with the Hough Transform lines overlaid on top
    k = 0.05
    sigma = 4
    rel_threshold = 0.01

    harris_response = my_corner_harris(edges, k, sigma)
    corners = my_corner_peaks(harris_response, rel_threshold)
    num_corners = len(corners)
    print(f"Number of corners found: {num_corners}")

    ## Finding pictures (rectangles) formed by Harris Corners using the custom find_pictures function
    print(f"Finding and extracting pictures in {imgpath}..")
    pictures, rectangles = find_pictures(corners, img_original)

    ## Counting number of pictures found
    pictures_found = len(pictures)
    print(f"Number of pictures found in {imgpath}: {pictures_found}")

    ## In the case where the corners found are significantly more than the pictures found, there must be picture overlap
    # In this case, we will find the corners again after using the Hough Transform to detect lines
    if num_corners > 14*pictures_found:
        print(f"The number of corners found is significantly larger than the number of pictures found, so there must be picture overlap, or the input image is too noisy.")
        print(f"Corner detection with Hough transform will now be applied.")
        hough_transform_applied = True
        # Applying the Hough transform to detect lines
        d_rho = 0.1
        d_theta = np.pi / 360
        n = 240
        H, L, res = my_hough_transform(edges, d_rho, d_theta, n)

        # Overlaying the detected lines on the original image
        for rho, theta in L:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 10000*(-b))
            y1 = int(y0 + 10000*(a))
            x2 = int(x0 - 10000*(-b))
            y2 = int(y0 - 10000*(a))
            cv2.line(lines, (x1, y1), (x2, y2), (255, 0, 0), 1)

        k = 0.05
        sigma = 4
        rel_threshold = 0.24
        harris_response = my_corner_harris(lines, k, sigma)

        # Redefining my_corner_peaks to work with the Hough Transform lines
        # The change in the redefinition is the size of the maximum_filter. It is now 100.
        def my_corner_peaks_redef(harris_response, rel_threshold):
            # Applying relative threshold
            threshold = np.max(harris_response) * rel_threshold
            corner_mask = harris_response > threshold

            # Keeping only the strongest corner in a local neighborhood (configured by size parameter in maximum_filter)
            local_max = maximum_filter(harris_response, size=100)
            maxima_mask = (harris_response == local_max)
            
            # Combining the above masks
            corners_mask = corner_mask & maxima_mask

            # Getting and returning the coordinates of the corners
            corner_locations = np.argwhere(corners_mask)
            return corner_locations

        corners = my_corner_peaks_redef(harris_response, rel_threshold)
        num_corners = len(corners)
        print(f"Number of corners found: {num_corners}")

        ## Finding pictures (rectangles) formed by Harris Corners using the custom find_pictures function
        print(f"Finding and extracting pictures in {imgpath}..")
        pictures, rectangles = find_pictures(corners, img_original)

    # Plotting the Hough Transform result if it was used.
    if hough_transform_applied == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(lines, cmap='gray', label = 'Lines from Hough Transform')
        plt.title(f'Hough Transform Result for {imgpath}')
        # plt.imsave(f'Report/Images/{base_name}_hough_transform.png', lines, cmap='gray')

    # Plotting detected corners and detected picture bounds on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_original)
    y, x = corners.T
    plt.scatter(x, y, color='red', s=10, label='Detected Corners')

    for i, rect in enumerate(rectangles):
        rect = np.array(rect)
        rect = rect[:, ::-1]  # Swap x and y
        if i == 0:  # add label only for the first rectangle to avoid duplicate legend entries
            plt.plot(*zip(*np.append(rect, [rect[0]], axis=0)), color='blue', label='Detected Picture Bounds')
        else:
            plt.plot(*zip(*np.append(rect, [rect[0]], axis=0)), color='blue')
    plt.title(f'Detected Corners and Picture Bounds in {base_name}')
    plt.legend()
    # plt.savefig(f'Report/Images/{base_name}_corners_bounds.png', bbox_inches='tight', pad_inches=0)


    ## Saving the extracted pictures
    # Defining output directory and creating it if it doesn't exist
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    # Get the file extension of the input image
    _, ext = os.path.splitext(imgpath)

    pictures_found_and_saved = 0
    for i, picture in enumerate(pictures):

        output_filename = os.path.join(output_dir, f'{base_name}_{i+1}{ext}')
        picture_to_save = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_filename, picture_to_save)
        print(f'Saved {output_filename}')
        plt.figure()
        plt.imshow(picture)
        plt.title(f'Extracted Picture {output_filename}')
        pictures_found_and_saved += 1

    print(f"Number of pictures found and saved in {imgpath}: {pictures_found_and_saved}")
    plt.show()

if __name__ == "__main__":
    my_lazy_scanner('im1.jpg')

    my_lazy_scanner('im2.jpg')

    my_lazy_scanner('im3.jpg')

    my_lazy_scanner('im4.jpg')

    my_lazy_scanner('im5.jpg')

