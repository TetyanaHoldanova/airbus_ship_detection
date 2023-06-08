import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from random import sample
from keras import utils, models

from image_segmentation.config import TEST_DIR, IMG_SCALING, RESULTS_DIR
from image_segmentation.utils.metrics import dice_p_bce, dice_coef
from image_segmentation.utils.rle import rle_encoder


def predict_test(img, model, test_path=TEST_DIR, img_scaling=IMG_SCALING):
    '''
    Preprocess and predict test images
    @param img: image id (str)
    @param model: unet model
    @param test_path: path to the test images (str)
    @param img_scaling: tuple with parameters for image scaling
    @return: np.array
    '''
    img = np.array(Image.open(test_path + img))         # Load image

    if pd.notna(img_scaling):                                
        img = img[::img_scaling[0], ::img_scaling[1]]   # Scale image if needed

    img = tf.expand_dims(img, axis=0)                   # Insert axis in np.array

    pred = model.predict(img)                           # Use model to detect ships
    pred = np.squeeze(pred, axis=0)                     # Delete axis 

    return pred


if __name__ == '__main__':

    utils.get_custom_objects()['dice_p_bce'] = dice_p_bce       # Specify custom metrics
    utils.get_custom_objects()['dice_coef'] = dice_coef

    model = models.load_model('assets/model/model.h5')          # Load model

    test_imgs = sample(os.listdir(TEST_DIR), 20)                # Take a sample of test images
    masks_rle = []

    for n, image_id in enumerate(test_imgs):
        mask = predict_test(image_id, model)                    # Returns a mask predicted for the test image
        masks_rle.append(rle_encoder(mask))                     # Encode the mask using Rle-encoder

    res = pd.DataFrame({'ImageId': test_imgs,                   # Write results in .csv file
                        'EncodedPixels': masks_rle})
    
    res.to_csv(RESULTS_DIR, index=False)

