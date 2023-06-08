import numpy as np
import pandas as pd
from PIL import Image

from image_segmentation.config import BATCH_SIZE, IMG_SCALING, TRAIN_DIR
from image_segmentation.utils.rle import masks_as_image


def data_generator(df, batch_size = BATCH_SIZE, img_scaling = IMG_SCALING):
    '''
    Make a python generator in order to get batches of training and validation samples
    @param df: training or validation dataframe
    @param batch_size: number of samples in one iteration
    '''
    # Group images and collect the corresponding masks
    unique_images = list(df.groupby('ImageId'))                       
    images = []
    masks = []
    
    while True:
        np.random.shuffle(unique_images)                              # Shuffle the images
        
        for img_id,  mask_df in unique_images:
            img = np.array(Image.open(TRAIN_DIR + img_id))            # Read the image file
            mask = masks_as_image(mask_df['EncodedPixels'].values)    # Make masks for the each image
            
            if pd.notna(img_scaling):                                 # Scale images and masks
                img = img[::img_scaling[0], ::img_scaling[1]]
                mask = mask[::img_scaling[0], ::img_scaling[1]]
    
            
            images.append(img.astype(np.float32)) 
            masks.append(mask.astype(np.float32))
                
            # Check if the lenght of the data more that batch size
            if len(images) >= batch_size:                              
                yield np.array(images) / 255.0, np.array(masks)       # Yield scaled images array and masks array
                images, masks = [], []                                # Сlean up images and masks arrays


def augmented_generator(generator, image_generator, masks_generator):
    '''
    Make a python generator in order to get augmented batches ofimages and masks
    @param generator: data generator
    @param image_generator: ImageDataGenerator object
    @param masks_generator: ImageDataGenerator object
    '''
    np.random.seed(42)
    
    # For generated batches of masks and images
    for images, masks in generator:
        # Generates batches of augmented images
        aug_imgs = image_generator.flow(255 * images,
                                        batch_size=len(images),
                                        seed=42, 
                                        shuffle=True)
        
        # Generates batches of augmented masks
        aug_masks = masks_generator.flow(masks,
                                        batch_size=len(masks),
                                        seed=42,
                                        shuffle=True)
        
        # Yield augmented images and masks 
        yield next(aug_imgs) / 255.0, next(aug_masks)


    