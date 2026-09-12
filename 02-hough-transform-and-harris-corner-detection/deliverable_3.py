# Dimitrios Ioannidis
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def rot_img(img, angle):

    # Getting the input image dimensions
    h, w = img.shape[:2]
    
    # Using the negative of the angle (counterclockwise rotation)
    angle = -angle
    
    # Computing the rotation matrix 
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Calculating the corners of the original image
    corners = np.array([
        [0, 0],
        [0, h],
        [w, 0],
        [w, h]
    ])

    # Translating corners to the origin, rotating and translating back
    corners_translated = corners - np.array([w / 2, h / 2])
    corners_rotated = (rotation_matrix @ corners_translated.T).T + np.array([w / 2, h / 2])

    # Calculating dimensions of the rotated image
    min_x, min_y = corners_rotated.min(axis=0)
    max_x, max_y = corners_rotated.max(axis=0)

    w_rot = int(np.ceil(max_x - min_x))
    h_rot = int(np.ceil(max_y - min_y))

    # Initializing the rotated image with a black background
    if len(img.shape) == 2:
        rotated_img = np.zeros((h_rot, w_rot), dtype=img.dtype)
    else:
        rotated_img = np.zeros((h_rot, w_rot, img.shape[2]), dtype=img.dtype)

    # Centers of the input and rotated image
    center_src = np.array([w / 2, h / 2])
    center_rot = np.array([w_rot / 2, h_rot / 2])

    # Creating a grid of coordinates in the rotated image
    x_rot, y_rot = np.meshgrid(np.arange(w_rot), np.arange(h_rot))

    # Calculating coordinates in the original image
    coords_translated = np.vstack([x_rot.ravel(), y_rot.ravel()]) - center_rot.reshape(-1, 1)
    coords_rotated = rotation_matrix.T @ coords_translated 
    coords_src = coords_rotated + center_src.reshape(-1, 1)

    x_src, y_src = coords_src

    # Creating a boolean mask for values within the image boundaries
    mask = (0 <= x_src) & (x_src < w) & (0 <= y_src) & (y_src < h)

    # Applying the mask to keep only valid coordinates
    x_src, y_src = x_src[mask], y_src[mask]

    # Flooring the source coordinates to get the top-left pixel
    x1, y1 = np.floor([x_src, y_src]).astype(int)

    # Ensuring the coordinates are within the image boundaries
    x2 = np.clip(x1 + 1, 0, w - 1)
    y2 = np.clip(y1 + 1, 0, h - 1)

    x_rot, y_rot = x_rot.ravel()[mask], y_rot.ravel()[mask]

    # Applying bilinear interpolation
    if len(img.shape) == 2:
        # Grayscale image case
        top_left = img[y1, x1]
        top_right = img[y1, x2]
        bottom_left = img[y2, x1]
        bottom_right = img[y2, x2]

        # Taking the average of the 4 pixels. Division is applied before addition to avoid overflow.
        rotated_img[y_rot, x_rot] = (top_left/4 + top_right/4+ bottom_left/4 + bottom_right/4)
    else:
        # RGB image case
        for c in range(img.shape[2]):
            top_left = img[y1, x1, c]
            top_right = img[y1, x2, c]
            bottom_left = img[y2, x1, c]
            bottom_right = img[y2, x2, c]

            # Taking the average of the 4 pixels. Division is applied before addition to avoid overflow.
            rotated_img[y_rot, x_rot, c] = (top_left/4 + top_right/4 + bottom_left/4 + bottom_right/4)

    return rotated_img


def deliverable_3():
    imgpath = 'im2.jpg'

    # Load the image
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    
    # Rotate the image by 54 degrees
    angle1 = 54 * np.pi / 180 # 54 degrees in radians
    tic1 = time.time()
    rotated_img1 = rot_img(img, angle1)
    toc1 = time.time()  
    print(f"Time taken to rotate the image by 54 degrees: {toc1 - tic1} seconds")

    # Rotate the image by 213 degrees
    angle2 = 213 * np.pi / 180 # 213 degrees in radians
    tic2 = time.time()
    rotated_img2 = rot_img(img, angle2)
    toc2 = time.time()
    print(f"Time taken to rotate the image by 213 degrees: {toc2 - tic2} seconds")
    
    # plt.imsave('Report/Images/rotated_image_54_degrees_deliverable3.jpg', cv2.cvtColor(rotated_img1, cv2.COLOR_BGR2RGB))
    # plt.imsave('Report/Images/rotated_image_213_degrees_deliverable3.jpg', cv2.cvtColor(rotated_img2, cv2.COLOR_BGR2RGB))

    # Display the original and rotated images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title('Rotated Image 54 degrees')
    plt.imshow(cv2.cvtColor(rotated_img1, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 3)
    plt.title('Rotated Image 213 degrees')
    plt.imshow(cv2.cvtColor(rotated_img2, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    deliverable_3()

