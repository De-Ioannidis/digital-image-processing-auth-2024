import numpy as np
from scipy.ndimage import convolve
from scipy import fft
import hw3_helper_utils
import matplotlib.pyplot as plt
from wiener_filtering import my_wiener_filter
import cv2
import os

def demo_script(x, noise_lvl, motion_blur_length, motion_blur_angle, imgname):
    print(f"Running demo on {imgname} with noise level {noise_lvl}, motion blur filter length {motion_blur_length}, and motion blur angle {motion_blur_angle}")
    v = noise_lvl * np.random.randn(*x.shape) # noise in the shape of the image

    h = hw3_helper_utils.create_motion_blur_filter(length=motion_blur_length, angle=motion_blur_angle) # motion blur filter

    y0 = convolve(x, h, mode="wrap") # blurred, noiseless image
    y0 = np.clip(y0, 0, 1)

    y = y0 + v # blurred, noisy image
    y = np.clip(y, 0, 1) # clip to [0, 1]

    # Finding the K that minimizes the error abs(x - x_hat)
    K_values = np.linspace(0.001, 30, 200)
    J = np.zeros_like(K_values)
    min_MSE = np.inf
    optimal_K = None
    optimal_x_hat = None

    for i, K in enumerate(K_values):
        x_hat = my_wiener_filter(y, h, K) # Wiener filtering

        J[i] = np.sum(np.abs(x - x_hat)**2)
        if J[i] < min_MSE:
            min_MSE = J[i]
            optimal_K = K
            optimal_x_hat = x_hat

    print(f"Optimal K: {optimal_K}")
    print(f"Minimum error: {min_MSE}\n\n")
    x_hat = optimal_x_hat   

    # Computing the Fourier Transform of h
    H = fft.fft2(h, s=x.shape)
    H_inv = np.zeros_like(H)
    H_abs = np.abs(H)
    H_inv[H_abs > 1e-6] = 1 / H[H_abs > 1e-6]  # to avoid division by zero or very small values

    # Computing x_inv and x_inv0
    x_inv = np.real(fft.ifft2(fft.fft2(y) * H_inv))
    x_inv0 = np.real(fft.ifft2(fft.fft2(y0) * H_inv))




    # -------------------------------------------------------
    ## PLOTTING
    # if not os.path.exists(f"figs"):
    #     os.makedirs(f"figs")

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    plt.suptitle(f"Noise level: {noise_lvl}, Motion blur filter length: {motion_blur_length}, Motion blur angle: {motion_blur_angle}, "
                 f"Optimal K: {optimal_K}, Minimum error: {min_MSE}")
    axs[0][0].imshow(x, cmap='gray')
    axs[0][0].set_title("Original image x")
    axs[0][1].imshow(y0, cmap='gray')
    axs[0][1].set_title("Blurred image y0")
    axs[0][2].imshow(y, cmap='gray')
    axs[0][2].set_title("Blurred and noisy image y")
    axs[1][0].imshow(x_inv0, cmap='gray')
    axs[1][0].set_title("Inverse filtering noiseless output x_inv0")
    axs[1][1].imshow(x_inv, cmap='gray')
    axs[1][1].set_title("Inverse filtering noisy output x_inv")
    axs[1][2].imshow(x_hat, cmap='gray')
    axs[1][2].set_title("Wiener filtering output x_hat")

    plt.tight_layout()


    plt.figure(figsize=(10, 6))  
    plt.scatter(optimal_K, min_MSE, color='r', marker='x', s=100, label='Chosen K')  # Plot chosen K as a red X
    plt.plot(K_values, J, label='Mean Square Error (MSE) vs. K', marker='.')  # Plot J vs. K
    plt.xlabel('K')  
    plt.ylabel('Mean Square Error (MSE)') 
    plt.title('MSE as a Function of K') 
    plt.legend()  
    plt.grid(True)
    plt.tight_layout() 
    # plt.savefig(f'Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_MSE_vs_K.png')  # Save the plot as a PNG file with 300 dpi
    # Check if output_images directory exists and create it if it doesn't
    if not os.path.exists("output_images"):
        os.makedirs("output_images")

    # # Save images FOR REPORT
    # if not os.path.exists(f"Report/images/{imgname}"):
    #     os.makedirs(f"Report/images/{imgname}")
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x.png", x, cmap='gray')
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_y0.png", y0, cmap='gray')
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_y.png", y, cmap='gray')
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_inv0.png", x_inv0, cmap='gray')
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_inv.png", x_inv, cmap='gray')
    # plt.imsave(f"Report/images/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_hat.png", x_hat, cmap='gray')

    # Save images
    imgname = imgname.split('.')[0]
    
    if not os.path.exists(f"assets/{imgname}"):
        os.makedirs(f"assets/{imgname}")

    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x.png", x, cmap='gray')
    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_y0.png", y0, cmap='gray')
    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_y.png", y, cmap='gray')
    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_inv0.png", x_inv0, cmap='gray')
    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_inv.png", x_inv, cmap='gray')
    plt.imsave(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_x_hat.png", x_hat, cmap='gray')
    plt.savefig(f"assets/{imgname}/noise_{noise_lvl}_motion_blur_{motion_blur_length}_angle_{motion_blur_angle}_MSE_vs_K.png")  


if __name__ == "__main__":
    # Loading image
    imgname = 'cameraman.tif'
    x = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE) / 255.0

    demo_script(x, 0.02, 10, 0, imgname) # noise level 0.02, motion blur filter length 10, angle 0
    demo_script(x, 0.02, 20, 30, imgname) # noise level 0.02, motion blur filter length 20, angle 30
    demo_script(x, 0.2, 10, 0, imgname) # noise level 0.2, motion blur filter length 10, angle 0
    demo_script(x, 0.2, 20, 30, imgname) # noise level 0.2, motion blur filter length 20, angle 30

    imgname = 'checkerboard.tif'
    x = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE) / 255.0
    demo_script(x, 0.02, 10, 0, imgname) # noise level 0.02, motion blur filter length 10, angle 0
    demo_script(x, 0.02, 20, 30, imgname) # noise level 0.02, motion blur filter length 20, angle 30
    demo_script(x, 0.2, 10, 0, imgname) # noise level 0.2, motion blur filter length 10, angle 0
    demo_script(x, 0.2, 20, 30, imgname) # noise level 0.2, motion blur filter length 20, angle 30

    plt.show()