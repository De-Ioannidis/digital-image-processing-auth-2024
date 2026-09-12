import numpy as np
from scipy import fft

def my_wiener_filter(y, h, K):
    # Dimensions of the input image and the impulse response
    M, N = y.shape
    L, P = h.shape

    # Zero-padding the impulse response to match the dimensions of y
    h_padded = np.pad(h, ((0, M - L), (0, N - P)), 'constant')

    # Computing the Fourier Transform of y and h
    Y = fft.fft2(y)
    H = fft.fft2(h_padded)

    # Computing the Wiener filter in the frequency domain
    H_conj = np.conj(H)
    H_abs = np.abs(H) ** 2
    W = H_conj / (H_abs + 1/K)

    # Applying the Wiener filter to Y
    X_hat = W * Y

    # Computing the inverse Fourier Transform to get the reconstructed image
    x_hat = np.real(fft.ifft2(X_hat))

    return x_hat