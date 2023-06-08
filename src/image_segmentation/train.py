import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

from image_segmentation.config import * 
from image_segmentation.utils.dataset import load_dataset
from image_segmentation.utils.generator import data_generator, augmented_generator
from image_segmentation.model.unet import Unet


if __name__ == '__main__':

    train_df, val_df = load_dataset()                 # Load dataset

    train_gen = data_generator(train_df)              # Make data generators for train and validation sets
    val_gen = data_generator(val_df)

    train_x, train_y = next(train_gen)         
    val_x, val_y = next(val_gen)

    params = {'rotation_range': 45,                   # Parameters for augmentation
            'horizontal_flip': True,
            'vertical_flip': True,                    
            'data_format': 'channels_last'} 

    image_generator = ImageDataGenerator(**params)    
    masks_generator = ImageDataGenerator(**params)

    gc.collect()                                      # Garbage collection
    
    unet = Unet()                                    
    model = unet.build(train_x.shape[1:])             # Build the Unet model
    
    # Train the model
    while True:
        loss_history = unet.fit(model, train_df, val_x, val_y, image_generator, masks_generator)
        if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.2:
            break

    model.save_weights('assets/model/model_weight.h5')
    model.save('assets/model/model.h5')
    
    gc.collect()                                      # Garbage collection
