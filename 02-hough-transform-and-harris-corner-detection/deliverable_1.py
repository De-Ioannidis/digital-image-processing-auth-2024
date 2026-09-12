# Dimitrios Ioannidis
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import numpy as np
import time

def my_hough_transform(img_binary, d_rho, d_theta, n):
    print("Computing Hough Transform..")


    ## Initializing the Hough Space
    N2, N1 = img_binary.shape
    rho_max = np.sqrt(N1 ** 2 + N2 ** 2)
    rho_range = np.arange(0, rho_max, d_rho)
    theta_range = np.arange(0, 2 * np.pi, d_theta)
    H = np.zeros((len(rho_range), len(theta_range)))


    ## Filling the Hough Space

    # Getting the indices of the edge points
    edge_points = np.column_stack(np.where(img_binary))

    # Calculating rho for all edge points and theta values using numpy 
    theta = theta_range[np.newaxis, :]  # shape (1, len(theta_range))
    n1, n2 = edge_points[:, 1][:, np.newaxis], edge_points[:, 0][:, np.newaxis]  # shape (num_edges, 1)
    rho = n1 * np.cos(theta) + n2 * np.sin(theta)  # shape (num_edges, len(theta_range))

    # Finding rho indices that are within the range of rho values
    rho_idx = (rho / d_rho).astype(np.int32)
    mask = (rho_idx >= 0) & (rho_idx < len(rho_range))

    # Updating the Hough accumulator array for valid rho_idx values
    for i in range(rho_idx.shape[0]):
        for j in range(rho_idx.shape[1]):
            if mask[i, j]:
                H[rho_idx[i, j], j] += 1


    ## Finding the n strongest lines in the Hough space

    H_flattened = H.flatten()  # flattening the Hough space matrix

    # Finding the indices of the n largest elements of H_flattened
    indices = np.argpartition(H_flattened, -n)[-n:]

    # Sorting the indices of the n largest elements in H_flattened
    indices = indices[np.argsort(H_flattened[indices])][::-1]

    # Converting the indices to the corresponding indices in H
    H_peaks_indices = np.column_stack(np.unravel_index(indices, H.shape))

    # Getting the corresponding rho and theta values
    rho_values = rho_range[H_peaks_indices[:, 0]]
    theta_values = theta_range[H_peaks_indices[:, 1]]

    # Combining the rho and theta values into a single array
    L = np.column_stack((rho_values, theta_values))


    ## Counting the number of edge points that are not on any of the n strongest lines

    # Calculate rho for each edge point for each of the strongest lines' theta values
    rho = n1 * np.cos(L[:, 1]) + n2 * np.sin(L[:, 1])

    # Calculate the minimum absolute difference between rho and all rho values of the strongest lines for each edge point
    min_diff = np.min(np.abs(rho.T - L[:, 0, np.newaxis]), axis=0)

    # Counting the number of edge points where the minimum absolute difference is greater than d_rho
    res = np.sum(min_diff > d_rho)

    return H, L, res

if __name__ == "__main__":

    imgpath = 'im2.jpg'

    # Loading the image
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

    # Converting the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img_gray = cv2.GaussianBlur(img_gray, (35, 35), 0)
    # Applying a binary threshold to the image
    # Pixels with a value greater than 235 are set to 255, all other pixels are set to 0
    _, img_binary = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY)

    # Applying a Gaussian filter to the binary image to get rid of noise
    img_binary_filtered = cv2.GaussianBlur(img_binary, (25,25), 0)

    plt.figure()
    plt.imshow(img_binary_filtered, cmap='gray')
    plt.title(f'Thresholding Result for {imgpath}')
    # plt.imsave('Report/Images/thresholding_deliverable1.png', img_binary_filtered, cmap='gray')

    # Applying the Canny edge detector
    edges = cv2.Canny(img_binary_filtered, 100, 200)

    plt.figure()
    plt.imshow(edges, cmap='gray')
    plt.title(f'Edge Detection Result for {imgpath}')
    # plt.imsave('Report/Images/edge_detection_deliverable1.png', edges, cmap='gray')

    # Print amount of edge points in edges
    print(f"Amount of edge points: {np.sum(edges/255).astype(int)}")

    # Computing the Hough transform and the L matrix using my_hough_transform
    d_rho = 0.1
    d_theta = np.pi / 360
    n = 240
    # Start the timer
    tic = time.time()

    H, L, res = my_hough_transform(edges, d_rho, d_theta, n)

    # Stop the timer and print the elapsed time
    toc = time.time()
    print(f"Time taken by my_hough_transform: {toc - tic} seconds")

    print(f"Amount of points that aren't on any of the {n} strongest lines: {res}")

    ## Plotting the image in Hough space (logarithmic scale)
    plt.figure(figsize=(10, 10))
    theta_range = np.arange(0, 2 * np.pi, d_theta)
    rho_range = np.arange(0, np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2), d_rho)
    H_log = np.log1p(H)
    plt.imshow(H_log, cmap='gray', aspect='auto', extent=[theta_range[0], theta_range[-1], rho_range[-1], rho_range[0]]) # aspect='auto', 
    plt.title(f'Hough Transform of {imgpath} (logarithmic)')
    plt.xlabel('Theta')
    plt.ylabel('Rho')
    colorbar = plt.colorbar(label='Votes')
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:0.0f}'.format(np.exp(x) - 1)))
    # Convert rho and theta values to row and column indices
    row_indices = L[:, 0]
    column_indices = L[:, 1]
    scatter = plt.scatter(column_indices, row_indices, color='red', s = 5)  # mark the points in L
    plt.legend([scatter], [f'L: {n} strongest lines'], loc='upper right')
    # Invert y-axis
    plt.gca().invert_yaxis()
    # plt.savefig('Report/Images/hough_transform_deliverable1.png', bbox_inches='tight', pad_inches=0)


    ## Plotting the lines from the Hough Transform on the original image
    plt.figure(figsize=(10, 10))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Plot the detected lines
    for rho, theta in L:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(img_rgb)
    plt.title(f'Detected Lines on {imgpath}')

    # Set the limits of the x and y axes to the size of the image
    plt.xlim([0, edges.shape[1]])
    plt.ylim([edges.shape[0], 0])
    # plt.imsave('Report/Images/detected_lines_deliverable1.jpg', img_rgb)
    
    plt.show()
