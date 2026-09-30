import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt

import time
import math
import os

from pathlib import Path

import skimage.io as skio 
from skimage import img_as_float
from skimage import data, transform

from scipy.signal import convolve2d

import matplotlib.pyplot as plt
from align_image_code import align_images


HERE = Path(__file__).parent  
MEDIA = HERE / '../../media'

#Kernels
box = np.ones((9, 9)) / 81
Dx = np.array([[1, 0, -1]], dtype=float)
Dy = np.array([[1], [0], [-1]], dtype=float)

def gaussian(sigma):
    ksize = int(6 * sigma + 1) | 1
    g = cv.getGaussianKernel(ksize, sigma)
    return g @ g.T

#Convolution Functions
def convolve_4_loop(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    h_pad = k_h//2
    w_pad = k_w//2

    flipped_kernel = np.flip(kernel)
    padded_image = np.pad(
        image,
        pad_width=((h_pad, h_pad), (w_pad, w_pad)),
        mode='constant',
        constant_values=0
    )

    output_image = np.zeros((i_h, i_w))
    for r in range(0, i_h):
        for c in range(0, i_w):
            # For each real pixel
            pixel_sum = 0
            for i in range(-h_pad, h_pad + 1):
                for j in range(-w_pad, w_pad + 1):
                    pixel_sum += padded_image[r + h_pad + i][c + w_pad + j] * flipped_kernel[i + h_pad][j + w_pad]
            output_image[r][c] = pixel_sum

    return output_image

def convolve_2_loop(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    h_pad = k_h//2
    w_pad = k_w//2

    flipped_kernel = np.flip(kernel)
    padded_image = np.pad(
        image,
        pad_width=((h_pad, h_pad), (w_pad, w_pad)),
        mode='constant',
        constant_values=0
    )

    output_image = np.zeros((i_h, i_w))
    for r in range(0, i_h):
        for c in range(0, i_w):
            # For each real pixel
            # Element wise multiply (padded_image slot * kernel) followed by summing all values

            output_image[r][c] = np.sum(padded_image[r: r + 2*h_pad + 1, c: c + 2*w_pad + 1] * flipped_kernel)
    
    return output_image

# Testing
def time_test():
    img = transform.resize(data.camera(), (128, 128), anti_aliasing=True)

    start = time.perf_counter()
    convolve_4_loop(img, box)
    print('4-loop:', time.perf_counter() - start, 's')

    start = time.perf_counter()
    convolve_2_loop(img, box)
    print('2-loop:', time.perf_counter() - start, 's')

    start = time.perf_counter()
    convolve2d(img, box, mode='same', boundary='fill', fillvalue=0)
    print('scipy: ', time.perf_counter() - start, 's')

def apply_filter(img, kernel, conv_fn=None):
    # Grayscale image
    if img.ndim == 2:
        if conv_fn is not None:
            return conv_fn(img, kernel)
        return convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)

    # Color image
    channels = []
    for c in range(img.shape[2]):
        filtered = apply_filter(img[:, :, c], kernel, conv_fn)
        channels.append(filtered)
    return np.stack(channels, axis=2)

def show_image(img, title='', save_name=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    if save_name:
        plt.savefig(MEDIA / (save_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

def show_images(img1, img2, title1, title2, save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    for ax, img, title in zip(axes, (img1, img2), (title1, title2)):
        if img.min() < 0:   # signed (derivative-style): center zero
            m = np.abs(img).max()
            im = ax.imshow(img, cmap='bwr', vmin=-m, vmax=m)
            label = 'response (blue = negative, red = positive)'
        else:
            im = ax.imshow(img, cmap='gray')
            label = 'intensity'
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()
    if save_name:
        plt.savefig(MEDIA / (save_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

def compare_images(img1, img2, title1, title2, save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, img, title in zip(axes, (img1, img2), (title1, title2)):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    if save_name:
        plt.savefig(MEDIA / (save_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()
 
def show_thresholds(original, dx_img, dy_img, thresholds, save_file_name = None):
    magnitude_image = np.sqrt(dx_img**2 + dy_img**2)

    # Grid sized to fit however many thresholds are passed in
    cols = math.ceil(math.sqrt(len(thresholds)))
    rows = math.ceil(len(thresholds) / cols)

    fig = plt.figure(figsize=(4 * (cols + 2), 3.5 * rows))
    gs = fig.add_gridspec(rows, cols + 2)

    ax_big = fig.add_subplot(gs[:, :2])
    ax_big.imshow(original, cmap='gray')
    ax_big.set_title('original')
    ax_big.axis('off')

    for idx, t in enumerate(thresholds):
        ax = fig.add_subplot(gs[idx // cols, 2 + idx % cols])
        ax.imshow(magnitude_image > t, cmap='gray')
        ax.set_title(f't = {t:.2f}')
        ax.axis('off')

    plt.tight_layout()
    if save_file_name is not None:
        plt.savefig(MEDIA / (save_file_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return magnitude_image

def hybrid_image(im_high, im_low, sigma_high, sigma_low):
    G_high = gaussian(sigma_high)
    G_low = gaussian(sigma_low)
    high_freq = im_high - apply_filter(im_high, G_high)
    low_freq = apply_filter(im_low, G_low)
    hybrid = low_freq + high_freq
    return hybrid, high_freq, low_freq

def fft(img):
    im2d = img.mean(axis=2)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2d))))

def show_ffts(images, titles, save_name=None):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(fft(img), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    if save_name:
        plt.savefig(MEDIA / (save_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

def gaussian_stack(img, levels, sigma):
    G = gaussian(sigma)
    stack = []
    current_img = img
    stack.append(current_img)
    for level in range(levels-1):
        current_img = apply_filter(current_img, G)
        stack.append(current_img)
    return stack

def laplacian_stack(img, levels, sigma):
    g_stack = gaussian_stack(img, levels, sigma)
    l_stack = []
    for g in range(len(g_stack) - 1):
        l_stack.append(g_stack[g] - g_stack[g + 1])
    l_stack.append(g_stack[-1])
    return l_stack

def blender(im1, im2, mask, levels, sigma):
    lap1 = laplacian_stack(im1, levels, sigma)
    lap2 = laplacian_stack(im2, levels, sigma)
    mask_g = gaussian_stack(mask, levels, sigma)
    mask1 = []
    mask2 = []
    blend = []
    for i in range(levels):
        curr_1_mask = mask_g[i] * lap1[i]
        curr_2_mask = (1 - mask_g[i]) * lap2[i]
        mask1.append(curr_1_mask)
        mask2.append(curr_2_mask)
        blend.append(curr_1_mask + curr_2_mask)

    result = np.clip(sum(blend), 0, 1)
    return result, mask1, mask2, blend

def show_fig342(m1, m2, bl, save_name=None):
    def norm(img):
        return (img - img.min()) / (img.max() - img.min())

    rows = []
    for lvl in [0, 2, 4]:
        rows.append([norm(m1[lvl]), norm(m2[lvl]), norm(bl[lvl])])
    rows.append([np.clip(sum(m1), 0, 1), np.clip(sum(m2), 0, 1), np.clip(sum(bl), 0, 1)])

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    for r in range(4):
        for c in range(3):
            axes[r, c].imshow(rows[r][c])
            axes[r, c].axis('off')

    plt.tight_layout()
    if save_name:
        plt.savefig(MEDIA / (save_name + '.png'), dpi=150, bbox_inches='tight')
    plt.show()

#Submissions
def p1_1_selfies():
    # Image of me (selfie)
    selfie_img = skio.imread('2/media/selfie.jpeg', as_gray=True)
    selfie_img = transform.rescale(selfie_img, 0.15, anti_aliasing=True)
    selfie_img = np.rot90(selfie_img, k=-1)

    selfie_box = apply_filter(selfie_img, box, convolve_2_loop)
    selfie_dx = apply_filter(selfie_img, Dx, convolve_2_loop)
    selfie_dy = apply_filter(selfie_img, Dy, convolve_2_loop)

    show_images(selfie_img, selfie_box, 'original', 'box', 'p1_1_box')
    show_images(selfie_img, selfie_dx, 'original', 'dx', 'p1_1_dx')
    show_images(selfie_img, selfie_dy, 'original', 'dy', 'p1_1_dy')

def p1_2_difference():
    # Image of Cameraman
    camera_man = img_as_float(data.camera())
    cman_dx = apply_filter(camera_man, Dx)
    cman_dy = apply_filter(camera_man, Dy)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    show_thresholds(camera_man, cman_dx, cman_dy, thresholds, 'p1_2_threshold_grid')
    show_thresholds(camera_man, cman_dx, cman_dy, [0.30], 'p1_2_threshold_030')

def p1_3_gaussian(sigma):
    G = gaussian(sigma)
    cman = img_as_float(data.camera())

    # Two Step
    gaussian_cman = apply_filter(cman, G)

    blur_dx = apply_filter(gaussian_cman, Dx)
    blur_dy = apply_filter(gaussian_cman, Dy)

    mag_blur = np.sqrt(blur_dx**2 + blur_dy**2)

    # Difference of Gaussians One Shot
    DoG_x = convolve2d(G, Dx)
    DoG_y = convolve2d(G, Dy)

    DoG_dx = apply_filter(cman, DoG_x)
    DoG_dy = apply_filter(cman, DoG_y)

    mag_DoG = np.sqrt(DoG_dx**2 + DoG_dy**2)

    # show_images(blur_dx, DoG_dx, 'blur_x', 'DoG_x', 'p1_3_blur_vs_DoG_dx')
    # show_images(blur_dy, DoG_dy, 'blur_y', 'DoG_y', 'p1_3_blur_vs_DoG_dy')
    # show_images(mag_blur, mag_DoG, 'mag_blur', 'mag_DoG', 'p1_3_blur_vs_DoG_mag')

    # show_images(DoG_x, DoG_y, 'DoG_x filter', 'DoG_y filter', 'p1_3_DoG_filters')
    # show_images(cman, gaussian_cman, 'original', 'Gaussian blur', 'p1_3_blur')

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    show_thresholds(cman, DoG_dx, DoG_dy, thresholds, 'p1_3_gaussian_9_thresholds')
    show_thresholds(cman, DoG_dx, DoG_dy, [0.10, 0.11, 0.12, 0.13, 0.14, 0.15], 'p1_3_gaussian_6_thresholds')
    show_thresholds(cman, DoG_dx, DoG_dy, [0.10], 'p1_3_gaussian_threshold_010')

    print('full match:', np.allclose(mag_blur, mag_DoG))
    print('interior match:', np.allclose(mag_blur[10:-10, 10:-10], mag_DoG[10:-10, 10:-10]))

def p2_1_sharpen(alpha, sigma):
    G = gaussian(sigma)
    delta = np.zeros_like(G)
    delta[G.shape[0] // 2, G.shape[1] // 2] = 1
    unsharp = (1 + alpha) * delta - alpha * G

    #Taj
    taj = skio.imread('2/media/taj.jpg', as_gray=True)

    blurred_taj = apply_filter(taj, G)
    show_image(blurred_taj, 'blurred', 'p2_1_blurred_taj')

    high_taj = taj - blurred_taj
    show_image(high_taj + 0.5, 'high-frequency', 'p2_1_highfreq_taj')

    sharp_taj = taj + alpha * high_taj
    show_image(sharp_taj, 'sharpened', 'p2_1_sharpened_taj')

    extra_sharp_taj = taj + 2 * alpha * high_taj
    show_image(extra_sharp_taj, 'extra-sharp', 'p2_1_extra_sharp_taj')

    unsharp_taj = apply_filter(taj, unsharp)
    compare_images(sharp_taj, unsharp_taj, 'sharp', 'unsharp', 'p2_1_unsharp_taj')

    #Stanley Cup
    cup = skio.imread('2/media/kings_cup.jpg', as_gray=True)

    blurred_cup = apply_filter(cup, G)
    show_image(blurred_cup, 'blurred', 'p2_1_blurred_cup')

    high_cup = cup - blurred_cup
    show_image(high_cup + 0.5, 'high-frequency', 'p2_1_highfreq_cup')

    sharp_cup = cup + alpha * high_cup
    show_image(sharp_cup, 'sharpened', 'p2_1_sharpened_cup')

    resharp_cup = apply_filter(blurred_cup, unsharp)
    compare_images(cup, resharp_cup, 'original', 're-sharpened', 'p2_1_blur_sharpened_cup')

def p2_2_hybrid(high_path, low_path, sigma_high, sigma_low, save_name):
    im1 = plt.imread(MEDIA / high_path) / 255.
    im2 = plt.imread(MEDIA / low_path) / 255.
    im1_aligned, im2_aligned = align_images(im1, im2)

    hybrid, high, low = hybrid_image(im1_aligned, im2_aligned, sigma_high, sigma_low)
    hybrid = np.clip(hybrid, 0, 1)

    compare_images(im1_aligned, im2_aligned, 'high_source', 'low_source', f'{save_name}_aligned')
    show_image(hybrid, 'hybrid', f'{save_name}_hybrid')

    return im1_aligned, im2_aligned, high, low, hybrid

#im1_aligned, im2_aligned, high, low, hybrid = p2_2_hybrid('./DerekPicture.jpg', './nutmeg.jpg', 4, 8, 'p2_2_derek_nutmeg')
# show_ffts([im1_aligned, im2_aligned, high, low, hybrid],
#           ['Derek (input)', 'Nutmeg (input)', 'high-pass', 'low-pass', 'hybrid'],
#           'p2_2_derek_nutmeg_fft')
#im1_aligned, im2_aligned, high, low, hybrid = p2_2_hybrid('./roommate.jpg', './grumpy_cat.jpg', 4, 23, 'p2_2_roommate_grumpy_cat')
#im1_aligned, im2_aligned, high, low, hybrid = p2_2_hybrid('./trump.jpg', './clown.jpg', 5, 5, 'p2_2_trump_clown')

def p2_4_blend(path1, path2, mask, levels, sigma, save_name):
    im1 = img_as_float(plt.imread(MEDIA / path1))[:, :, :3]
    im2 = img_as_float(plt.imread(MEDIA / path2))[:, :, :3]
    im2 = transform.resize(im2, im1.shape[:2], anti_aliasing=True)

    if mask is None:
        h, w = im1.shape[:2]
        mask = np.zeros((h, w, 3))
        mask[:, :w //2] = 1

    result, masked1, masked2, blended = blender(im1, im2, mask, levels, sigma)

    compare_images(im1, im2, 'image 1', 'image 2', save_name + '_inputs')
    show_image(result, 'blended', save_name + '_result')
    return im1, im2, mask, result, masked1, masked2, blended

# im1, im2, mask, result, m1, m2, bl = p2_4_blend('apple.jpeg', 'orange.jpeg', None, 6, 10, 'p2_4_orange_apple')
# show_fig342(m1, m2, bl, 'p2_3_fig')

#im1, im2, mask, result, m1, m2, bl = p2_4_blend('puff.jpg', 'kirby.jpg', None, 4, 5, 'p2_4_puffirby')

sf = img_as_float(plt.imread(MEDIA / 'sf.jpeg'))[:, :, :3]
h, w = sf.shape[:2]
yy, xx = np.mgrid[0:h, 0:w]
circle = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.3 * min(h, w)) ** 2
mask = np.dstack([circle, circle, circle]).astype(float)

im1, im2, mask, result, m1, m2, bl = p2_4_blend('sf.jpeg', 'la.jpg', mask, 4, 5, 'p2_4_safl')