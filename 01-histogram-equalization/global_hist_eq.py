# Dimitrios Ioannidis
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_equalization_transform_of_img(img_array: np.ndarray):
    # We assume that the input image is an 8-bit grayscale image
    histogram = compute_histogram(img_array) # compute the image's histogram
    cdf = np.cumsum(histogram) # compute the image's cumulative distribution function
    cdf_normalized = cdf / cdf.max() # normalize it to [0, 1]

    equalization_transform = np.round(cdf_normalized * 255) # from theory: y_k = round(cdf_k * L-1)

    return equalization_transform

def perform_global_hist_equalization(img_array: np.ndarray):
    equalization_transform = get_equalization_transform_of_img(img_array)

    equalized_img_array_flat = equalization_transform[img_array.flatten()] # indexing the equalization transform with the img_array. This will map each pixel value to its equalized value.
    equalized_img_array = equalized_img_array_flat.reshape(img_array.shape) # reshape the flat array back to the original image shape

    return equalized_img_array


# Function to compute the histogram of an image - returns an array with the number of pixels that have each intensity value 
def compute_histogram(img_array):
    histogram = np.zeros(256)

    for pixel in img_array.flatten():
        # Increment the count for the current pixel value
        histogram[int(pixel)] += 1

    return histogram






# ----------------------------------------------------------------------------------------------------------------------------------







# FOR TESTING
if __name__ == "__main__":
    # Set the filepath to the image file
    filename = "input_img.png"
    # Read image
    img = Image.open(fp=filename)
    
    # Keep only the Luminance component of the image
    # This converts the image to an 8-bit grayscale image (256 different luminance values)
    bw_img = img.convert("L")
    img_array = np.array(bw_img)

    # Apply the equalization transform to the image
    equalized_img_array = perform_global_hist_equalization(img_array)

    # Compute the histograms
    histogram = compute_histogram(img_array)
    cdf = np.cumsum(histogram)
    equalized_histogram = compute_histogram(equalized_img_array)
    equalized_cdf = np.cumsum(equalized_histogram)

    bins = np.arange(256) # bins for the histogram plotting
    # Plot the original and equalized histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Global Histogram Equalization')

    axs[0, 0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('Original Image')
    axs[0, 1].bar(bins, histogram, width=1, label='Histogram')
    axs[0, 1].set_title('Original Histogram')
    axs[0, 2].bar(bins, histogram/histogram.max(), width=1, label='Normalized Histogram')
    axs[0, 2].plot(bins, cdf/cdf.max(), color='r', label='CDF')
    axs[0, 2].set_title('Normalized Original Histogram and CDF')
    axs[0, 2].legend()

    axs[1, 0].imshow(equalized_img_array, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title('Equalized Image')
    axs[1, 1].bar(bins, equalized_histogram, width=1, label='Histogram')
    axs[1, 1].set_title('Equalized Histogram')
    axs[1, 2].bar(bins, equalized_histogram/equalized_histogram.max(), width=1, label='Normalized Histogram')
    axs[1, 2].plot(bins, equalized_cdf/equalized_cdf.max(), color='r', label='CDF')
    axs[1, 2].set_title('Normalized Equalized Histogram and CDF')
    axs[1, 2].legend()

    plt.show()


