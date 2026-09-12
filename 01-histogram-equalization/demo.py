# Dimitrios Ioannidis
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from global_hist_eq import perform_global_hist_equalization, get_equalization_transform_of_img, compute_histogram
from adaptive_hist_eq import perform_adaptive_hist_equalization, perform_ahe_without_interpolation


if __name__ == "__main__":
    # Set the filepath to the image file
    filename = "input_img.png"
    # Read image
    img = Image.open(fp=filename)
    
    # Keep only the Luminance component of the image
    # This converts the image to an 8-bit grayscale image (256 different luminance values)
    bw_img = img.convert("L")
    img_array = np.array(bw_img)

    # Apply the global histogram equalization to the image
    ghe_img_array = perform_global_hist_equalization(img_array)

    # Apply the adaptive histogram equalization to the image
    region_len_h = 48
    region_len_w = 64
    ahe_img_array = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)

    # Apply AHE at every region without interpolation
    ahe_no_interp_img_array = perform_ahe_without_interpolation(img_array, region_len_h, region_len_w)

    import os
    os.makedirs('assets', exist_ok=True)

    # Plotting the global equalization transform
    fig, ax = plt.subplots(figsize=(7, 5))
    equalization_transform = get_equalization_transform_of_img(img_array)
    ax.plot(equalization_transform)
    ax.set_title("Global Equalization Transform")
    plt.savefig('assets/global_equalization_transform.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img_array, cmap="gray")
    ax.set_title("Original Image")
    plt.savefig('assets/original_image.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(ghe_img_array, cmap="gray")
    ax.set_title("Globally Equalized Image")
    plt.savefig('assets/globally_equalized.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    hist = compute_histogram(img_array)
    ax[0].bar(np.arange(256), hist, width=1)
    ax[0].set_title("Original Image Histogram")
    hist = compute_histogram(ghe_img_array)
    ax[1].bar(np.arange(256), hist, width=1)
    ax[1].set_title("Globally Equalized Image Histogram")
    plt.savefig('assets/global_hist_comparison.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(ahe_img_array, cmap="gray")
    ax.set_title("Adaptively Equalized Image")
    plt.savefig('assets/adaptively_equalized.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    hist = compute_histogram(img_array)
    ax[0].bar(np.arange(256), hist, width=1)
    ax[0].set_title("Original Image Histogram")
    hist = compute_histogram(ahe_img_array)
    ax[1].bar(np.arange(256), hist, width=1)
    ax[1].set_title("Adaptively Equalized Image Histogram")
    plt.savefig('assets/adaptive_hist_comparison.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(ahe_no_interp_img_array, cmap="gray")
    ax.set_title("Image after AHE without Interpolation")
    plt.savefig('assets/ahe_no_interpolation.png', bbox_inches='tight')
    plt.show()


