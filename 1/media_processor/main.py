# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Helped Functions

def blur(channel, kernel_size, sigma_x = 0):
    assert kernel_size%2 == 1, "Kernel_Size must be odd"
    return cv.GaussianBlur(channel, (kernel_size, kernel_size), sigma_x)

def shift(channel, shift_dx, shift_dy):
    shifted_dx = np.roll(channel, shift=shift_dx, axis=1)
    shifted_dxdy = np.roll(shifted_dx, shift=shift_dy, axis=0)
    return shifted_dxdy

def crop(channel, margin):
    height, width = channel.shape
    return channel[int(height*margin): height - int(height*margin), int(width*margin): width - int(width*margin)]

def cut(channel):
    return channel[::2, ::2]

def L2_Norm_Score(channel, reference):
    return np.sum((channel-reference)**2)

def Normalized_Cross_Co_Score(channel, reference):
    sub_mean_channel = (channel - channel.mean())
    sub_mean_ref = (reference - reference.mean())
    sub_mean_norm_channel = sub_mean_channel / np.linalg.norm(sub_mean_channel)
    sub_mean_norm_ref = sub_mean_ref / np.linalg.norm(sub_mean_ref)
    return np.sum(sub_mean_norm_channel * sub_mean_norm_ref)

def locate_best_offset(channel, reference, window, margin, initial_dx = 0, initial_dy = 0, score_metric = "NCC"):
    """
    Inputs : 
        Channel & Reference - Color Channels
        Margin - Crop Amount
        Window - Max pixel displacement
        Initial_dx/dy - Origin of displacement
        Score_metric - Type of Test Used to evaluate effectiveness of shift
            
    Output : 
        Tuple (dx, dy) offset to align channel to refernce
    
    Repeatedly shifts image (based on window and initial displacements) and
    scores them based on score_metric as to how well the image shift performs.
    Saves score as best_score if it is better given the metric and saves the
    dx and dy that resulted in that scores

    Return the dx and dy at the end that gave the most optimal score.
    """
    best_dx = 0
    best_dy = 0
    best_score = 0
    if score_metric == "L2":
        best_score = float('inf')
    elif score_metric == "NCC":
        best_score = -float('inf')

    cropped_reference = crop(reference, margin)

    for dx in range(-window, window+1):
        for dy in range (-window, window+1):
            
            shifted_dxdy_cropped_channel = crop(shift(channel, dx + initial_dx, dy + initial_dy), margin)

            if score_metric == "L2":
                score = L2_Norm_Score(shifted_dxdy_cropped_channel, cropped_reference)
                if score < best_score:
                    best_score = score
                    best_dx = dx + initial_dx
                    best_dy = dy + initial_dy
            elif score_metric == "NCC":
                score = Normalized_Cross_Co_Score(shifted_dxdy_cropped_channel, cropped_reference)
                if score > best_score:
                    best_score = score
                    best_dx = dx + initial_dx
                    best_dy = dy + initial_dy
    return (best_dx, best_dy)

def align(channel, reference, margin, window, smallest_image_size = 0, score_metric = "NCC"):
    """
    Inputs: 
        Channel & Reference - Color Channels
        Margin - Crop Amount
        Window is the max pixel displacement
        Smallest_Image_Size - Pixel count
        
    Output : 
        Tuple (dx, dy) offset to align channel to refernce

    Align recursively (or in the base case doesn't) applies a gaussian blur, followed
    by a cut of even rows and columns, and applies align again. Once align reaches the 
    base case (height or width are < smallest_image size), it finds the best offset and passes it back. 
    -> (details for inputs to locate_best_off can be found its it's docstring)) 
    
    Then in the previous recursion, it checks the best offset given it should start 
    somewhere based on the deeper recursions (this recursions best_dx/best_dy), and 
    passes that value back up the chain. 
    
    Eventually, the best dx/dy pair is returned that (ideally) gives the best
    image shift based on the score_metric.

    """
    
    height, width = channel.shape

    if height <= smallest_image_size or width <= smallest_image_size:
    # Base Case, Lowest Depth
        result = locate_best_offset(channel, reference, 15, margin, score_metric = score_metric)
        #print(f"base case: {result}")
        return result
    else:
    # Higher Level (Using Pyramid)
        gaussian_channel = blur(channel, 5)
        gaussian_ref = blur(reference, 5)
        cut_channel = cut(gaussian_channel)
        cut_ref = cut(gaussian_ref)
        best_dx, best_dy = align(cut_channel, cut_ref, margin, window, smallest_image_size, score_metric = score_metric)
        result =  locate_best_offset(channel, reference, window, margin, best_dx*2, best_dy*2 ,score_metric)
        return result

# Main color image creation function
def save_img(im, path):
    # prepare for OpenCV saving/display (expects BGR uint8)
    out_uint8 = np.clip(im * 255.0, 0, 255).astype(np.uint8)
    out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

    # save the image
    cv.imwrite(path, out_bgr)

def color(imname, out_path_aligned, out_path_unaligned, margin, pyramid_window, smallest_pixel_dimension, score_metric = "NCC"):
    # read in the image as grayscale (the glass plate scan is stacked grayscale)
    im = cv.imread(imname, cv.IMREAD_GRAYSCALE)

    # convert to float in [0,1] (might want to do this later on to save memory)
    im = im.astype(np.float32) / 255.0

    # compute the height of each part (just 1/3 of total)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # create unaligned color image  
    im_unaligned = np.dstack([r, g, b])

    # obtain correct shifts and perform shifts
    g_dx, g_dy = align(g, b, margin, pyramid_window, smallest_pixel_dimension, score_metric)
    r_dx, r_dy = align(r, b, margin, pyramid_window, smallest_pixel_dimension, score_metric)

    print(f"File - {imname}\nAG: DX, DY - ", g_dx, g_dy, "\nAR: DX, DY - ", r_dx, r_dy)

    shifted_g = shift(g, g_dx, g_dy)
    shifted_r = shift(r, r_dx, r_dy)
    
    # create aligned color image  
    im_out = np.dstack([shifted_r, shifted_g, b])

    save_img(im_out, out_path_aligned)
    save_img(im_unaligned, out_path_unaligned)

#----------------------------------------------------#
# display the image using matplotlib (expects RGB)

# fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# axes[0].imshow(im_unaligned)
# axes[0].set_title('Unaligned')
# axes[0].axis('off')

# axes[1].imshow(im_out)
# axes[1].set_title('Aligned')
# axes[1].axis('off')

# plt.show()

names = ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg',
         'church.tif', 'emir.tif', 'harvesters.tif', 'icon.tif',
         'ilemselga.tif', 'melons.tif', 'religous_painting.tif',
         'self_portrait.tif', 'siren.tif', 'three_generations.tif',
         'wharf.tif', 'Bashenka.tif', 'Kafedra.tif', 'V Malorossii_Lake.tif', 
         'V Malorossii_Woman.tif']

names_2 = []

here = os.path.dirname(os.path.abspath(__file__))

out_dir = os.path.join(here, 'output')
os.makedirs(out_dir, exist_ok=True)

# name of the input file

for name in names:
    color(os.path.join(here, name), 
          os.path.join(out_dir, 'out_aligned_' + name.split('.')[0] + '.jpg'),
          os.path.join(out_dir, 'out_unaligned_' + name.split('.')[0] + '.jpg'),
          margin=0.15, 
          pyramid_window=5, 
          smallest_pixel_dimension=100)
