# Dimitrios Ioannidis
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, convolve


def convolution(img, kernel):
    return convolve(img, kernel)

def my_corner_harris(img, k, sigma):
    print("Computing Harris Corners..")

    ## Computing "A" matrix
    # Computing image gradients using Sobel operator
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
    # Computing products of derivatives
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    ## Creating Gaussian window
    size = round(4*sigma)
    size = size + 1 if size % 2 == 0 else size  # Ensure size is odd so that the window has a center pixel
    # Creating 1D Gaussian
    x = np.arange(-size // 2 + 1., size // 2 + 1.) 
    gauss = np.exp(-(x**2) / (2*sigma**2))
    gauss /= gauss.sum()  # normalize
    # Creating 2D Gaussian window using outer product
    window = np.outer(gauss, gauss)

    ## Creating "M" matrix 
    # Convolving products of derivatives with the window
    Sxx = convolution(Ixx, window)
    Syy = convolution(Iyy, window)
    Sxy = convolution(Ixy, window)

    # Computing the response of the Harris corner detector
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    harris_response = det - k*(trace**2)
    
    return harris_response


def my_corner_peaks(harris_response, rel_threshold):
    # Applying relative threshold
    threshold = np.max(harris_response) * rel_threshold
    corner_mask = harris_response > threshold

    # Keeping only the strongest corner in a local neighborhood (configured by size parameter in maximum_filter)
    img_diagonal_size = int(np.sqrt(harris_response.shape[0]**2 + harris_response.shape[1]**2))
    filter_size = img_diagonal_size // 20 - 105 # empirical value depending on image size
    local_max = maximum_filter(harris_response, size=filter_size)
    maxima_mask = (harris_response == local_max)
    
    # Combining the above masks
    corners_mask = corner_mask & maxima_mask

    # Getting and returning the coordinates of the corners
    corner_locations = np.argwhere(corners_mask)

    return corner_locations



if __name__ == "__main__":
    imgpath = 'im2.jpg'

    # Loading the image
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

    # Converting the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying a binary threshold to the image
    # Pixels with a value greater than 235 are set to 255, all other pixels are set to 0
    _, img_binary = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY)

    img_binary_filtered = cv2.GaussianBlur(img_binary, (27, 27), 0)

    # Apply the Canny edge detector
    edges = cv2.Canny(img_binary_filtered, 100, 200)

    print(f"Amount of edge points: {np.sum(edges/255).astype(int)}")

    # Parameters for the Harris corner detector
    k = 0.05
    sigma = 4
    rel_threshold = 0.01

    # Applying my_corner_harris and my_corner_peaks
    harris_response = my_corner_harris(edges, k, sigma)
    my_corners = my_corner_peaks(harris_response, rel_threshold)

    # Plotting the results
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    im1 = axes[0].imshow(harris_response, cmap='coolwarm')
    axes[0].set_title(f'Harris Response for {imgpath}')
    fig.colorbar(im1, ax=axes[0])

    axes[1].imshow(edges, cmap='gray')
    axes[1].plot(my_corners[:, 1], my_corners[:, 0], 'r.', markersize=2)
    axes[1].set_title('My Harris Corners')

    # plt.savefig('Report/Images/harris_corner_detection_deliverable2.png',  bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    plt.show()

