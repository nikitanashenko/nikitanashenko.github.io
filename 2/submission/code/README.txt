CS180 Project 2 - Fun with Filters and Frequencies
==================================================
 
FILES
-----
main.py               All code for Parts 1 and 2 (one function per part)
align_image_code.py   Provided starter code for aligning hybrid image pairs
                      (only change: ginput(2, timeout=0) so it waits for clicks)
 
Input images and all generated figures are in ../../media/
 
DEPENDENCIES
------------
Python 3, numpy, scipy, matplotlib, scikit-image, opencv-python
 
    pip install numpy scipy matplotlib scikit-image opencv-python
 
HOW TO RUN
----------
Uncomment the call(s) for the part you want at the bottom of main.py, then:
 
    python main.py
 
Each part saves its figures to ../../media/ (paths are relative to main.py,
so it works from any working directory).
 
    Part 1.1   p1_1_selfies()           convolution from scratch (4-loop and 2-loop)
               time_test()              runtime comparison vs scipy.signal.convolve2d
    Part 1.2   p1_2_difference()        finite differences, gradient magnitude, thresholds
    Part 1.3   p1_3_gaussian(1)         Gaussian blur + DoG filters, verification
    Part 2.1   p2_1_sharpen(1, 1)       unsharp masking (alpha, sigma)
    Part 2.2   p2_2_hybrid(...)         hybrid images (requires clicking 2 points
                                        per image, e.g. both eyes, same order)
               show_ffts(...)           log-magnitude FFTs for the hybrid
    Part 2.3   gaussian_stack / laplacian_stack, show_fig342(...)
    Part 2.4   p2_4_blend(...)          multiresolution blending
                                        (mask=None -> vertical seam)
 
KEY FUNCTIONS
-------------
convolve_4_loop, convolve_2_loop   numpy-only convolution, zero padding, "same" size
apply_filter                       scipy convolve2d wrapper; filters color images
                                   one channel at a time
gaussian(sigma)                    2D Gaussian via cv2.getGaussianKernel, size ~6*sigma+1
hybrid_image                       high-pass(im1) + low-pass(im2)
gaussian_stack, laplacian_stack    stacks without downsampling
blender                            blends Laplacian levels using a Gaussian stack of the mask
 
NOTES
-----
- Zero padding causes a thin border artifact in derivative / blurred images.
- Hybrid alignment rotates one image; the rotated frame edges can appear as
  faint lines in the hybrid and its FFT.