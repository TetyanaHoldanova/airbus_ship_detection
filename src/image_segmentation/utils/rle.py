import numpy as np
import pandas as pd

from image_segmentation.config import H, W


def rle_decoder(rle_string):
    '''
    Convert RLE-encoded values into the mask
    @param rle_string: str
    @return: np.array with shape (W, H)
    '''    
    # Return zero matrices if the image does not have ships
    if pd.isna(rle_string):
        return np.zeros((H, W))
    
    rle_string = [int(n) for n in rle_string.split(' ')]
    mask = np.zeros(H * W, dtype=np.uint8)   # Create a zero matrix as a background
    
    for i in range(0, len(rle_string), 2):
        start = rle_string[i] - 1            # Find the start position
        end = start + rle_string[i+1]        # Find the end position
        mask[start:end] = 1                  # Fill ship pixels with 1
        
    return mask.reshape(H, W).T


def masks_as_image(rle_list):
    '''
    Create full mask of the training image
    @param rle_list: list of the RLE-encoded masks of each ship in one whole training image
    @return: np.ndarray
    '''
    masks = np.zeros((768, 768), dtype = np.int16)     # Create a zero matrix as a background
    
    for mask in rle_list:                               
        if isinstance(mask, str): 
            masks += rle_decoder(mask)                 # Use rle_decoder to create mask for whole image
    
    return np.expand_dims(masks, -1)


def rle_encoder(mask_image):
    '''
    Rle-encoder for masks
    @param mask_image: np.array 
    @return: string/None
    '''
    pixels = mask_image.flatten()              # Reshape the array
    pixels[0] = 0                              # Setting corner pixels to 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2     # Identifying ship pixels, convert them to rle-format 
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle_str = ' '.join(str(x) for x in runs)              # Make a rle-string
    if len(rle_str) != 0:                                 
        return rle_str
    return None