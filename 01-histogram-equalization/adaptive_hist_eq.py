# Dimitrios Ioannidis
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from global_hist_eq import get_equalization_transform_of_img, compute_histogram 


def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    
    # Get the number of regions in the height and width directions. We use the // operator for floor division, since the number of regions needs to be an integer    
    num_regions_h = img_array.shape[0] // region_len_h # number of contextual regions in the height direction
    num_regions_w = img_array.shape[1] // region_len_w # number of contextual regions in the width direction

    # Initialize the dictionary to store the equalization transformations of each region
    equalization_transformations = {}

    # Iterating through all contextual regions
    for i in range(num_regions_h):
        for j in range(num_regions_w):
            # Get current contextual region
            contextual_region = img_array[i*region_len_h:(i+1)*region_len_h, j*region_len_w:(j+1)*region_len_w]
            
            # Use the get_equalization_transform_of_img function to get the equalization transform
            equalization_transform = get_equalization_transform_of_img(contextual_region)

            # Store equalization transformation in the dictionary with the tuple as the key
            equalization_transformations[(i*region_len_h, j*region_len_w)] = equalization_transform

    return equalization_transformations


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int, region_len_w: int):
    equalization_transformations = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    # Get the number of regions in the height and width directions. We use the // operator for floor division, since the number of regions needs to be an integer    
    num_regions_h = img_array.shape[0] // region_len_h
    num_regions_w = img_array.shape[1] // region_len_w

    equalized_img_array = np.copy(img_array) # initialize the AHE equalized image array

    for i in range(num_regions_h):
        for j in range(num_regions_w):
            
                # Iterating through all points in the current contextual region 
                for h_p in range(i*region_len_h, (i+1)*region_len_h):
                    for w_p in range (j*region_len_w, (j+1)*region_len_w):

                        x = img_array[h_p, w_p] # the pixel value of the point (h_p, w_p)

                        # Compute h_-, h_+, w_-, w_+ according to which quarter of the current contextual region the point (h_p, w_p) is in
                        h_minus = i*region_len_h + (region_len_h/2 if h_p > (i + 0.5)*region_len_h else -region_len_h/2)
                        w_minus = j*region_len_w + (region_len_w/2 if w_p > (j + 0.5)*region_len_w else -region_len_w/2)
                        h_plus = i*region_len_h + (region_len_h/2 if h_p <= (i + 0.5)*region_len_h else region_len_h + region_len_h/2)
                        w_plus = j*region_len_w + (region_len_w/2 if w_p <= (j + 0.5)*region_len_w else region_len_w + region_len_w/2)

                        # If the point is on the border of the image, the transformation is just the transformation of the contextual region
                        if h_minus < 0 or h_plus >= img_array.shape[0] or w_minus < 0 or w_plus >= img_array.shape[1]:
                            y = equalization_transformations[(i*region_len_h, j*region_len_w)][x]
                            equalized_img_array[h_p, w_p] = y
                        # Else, the point is not on the border of the image, use interpolation
                        else:
                            # Compute a and b
                            a = (w_p - w_minus) / region_len_w # a = (w_p - w_-) / (w_+ - w_-)
                            b = (h_p - h_minus) / region_len_h # b = (h_p - h_-) / (h_+ - h_-)

                            # Determine all possible indices for the equalization transformations
                            h_indices = [(i-1)*region_len_h, i*region_len_h, (i+1)*region_len_h]
                            w_indices = [(j-1)*region_len_w, j*region_len_w, (j+1)*region_len_w]

                            # Determine the indices of the equalization transformations to use for bilinear interpolation
                            # If we are in the upper left of the contextual region, then the transformations of the regions where the four closest contextual centers are located have indices ((i-1)*region_len_h, (j-1)*region_len_w), ((i-1)*region_len_h, j*region_len_w), (i*region_len_h, (j-1)*region_len_w), (i*region_len_h, j*region_len_w))
                            # So on for the other cases..
                            h_index = (h_indices[0], h_indices[1]) if h_p <= (i + 0.5)*region_len_h else (h_indices[1], h_indices[2])
                            w_index = (w_indices[0], w_indices[1]) if w_p <= (j + 0.5)*region_len_w else (w_indices[1], w_indices[2])

                            # Bilinear interpolation formula, using the correct indices as determined above
                            y = (1 - a) * (1 - b) * equalization_transformations[h_index[0], w_index[0]][x] + \
                                                a * (1 - b) * equalization_transformations[h_index[0], w_index[1]][x] + \
                                                (1 - a) * b * equalization_transformations[h_index[1], w_index[0]][x] + \
                                                a * b * equalization_transformations[h_index[1], w_index[1]][x]

                            equalized_img_array[h_p, w_p] = y # setting the equalized pixel value according to bilinear interpolation in the equalized image array
    
    return equalized_img_array




# Function to perform adaptive histogram equalization at every region without interpolation - used in demo.py
def perform_ahe_without_interpolation(img_array: np.ndarray, region_len_h: int, region_len_w: int):

    equalization_transformations = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    num_regions_h = img_array.shape[0] // region_len_h
    num_regions_w = img_array.shape[1] // region_len_w

    equalized_img_array = np.copy(img_array) # initialize the AHE equalized image array

    for i in range(num_regions_h):
        for j in range(num_regions_w):
            equalization_transform = equalization_transformations[(i*region_len_h, j*region_len_w)] # get equalization transform from equalization_transforms dictionary
            contextual_region = img_array[i*region_len_h:(i+1)*region_len_h, j*region_len_w:(j+1)*region_len_w] # getting the entire contextual region
            
            equalized_region_flat = equalization_transform[contextual_region.flatten()] # transforming the entire contextual region with its equalization transform
            equalized_region = equalized_region_flat.reshape(contextual_region.shape) # reshaping
            
            equalized_img_array[i*region_len_h:(i+1)*region_len_h, j*region_len_w:(j+1)*region_len_w] = equalized_region # setting the equalized region in the equalized image array

    return equalized_img_array






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

    # Define the region lengths
    region_len_h = 36
    region_len_w = 48

    # Apply the adaptive equalization transform to the image
    equalized_img_array = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)

    # Compute the histograms
    histogram = compute_histogram(img_array)
    cdf = np.cumsum(histogram)
    equalized_histogram = compute_histogram(equalized_img_array)
    equalized_cdf = np.cumsum(equalized_histogram)

    bins = np.arange(256) # bins for the histogram plotting
    # Plot the original and equalized histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Adaptive Histogram Equalization')

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
