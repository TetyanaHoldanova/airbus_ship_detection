import os
import pandas as pd
from sklearn.model_selection import train_test_split
from image_segmentation.config import TRAIN_DIR, SEGMENTATION_DIR, GROUP_SIZE


def load_dataset():
    '''
    Load, preprocess and split dataset
    @return: 2 x pd.DataFrame (n, 2)
    '''

    masks = pd.read_csv(SEGMENTATION_DIR)

    # Mark files that contain ship
    masks['has_ship'] = masks['EncodedPixels'].map(lambda x: 0 if type(x) == float else 1)
    
    # Count ships on each image
    unique_images = masks.groupby('ImageId').agg(n_ships=('has_ship', 'sum')).reset_index()
    masks.drop('has_ship', axis=1, inplace=True)
   
    # Delete corrupt files with size < 50 kB
    unique_images['file_size'] = unique_images['ImageId'].map(lambda x: os.stat(TRAIN_DIR + x).st_size/1024)
    unique_images = unique_images[unique_images['file_size'] > 50]
    unique_images.drop('file_size', axis=1, inplace=True)
   
    # Get balanced dataset
    balanced_df = unique_images.groupby('n_ships', group_keys=False).apply(lambda x: x.sample(GROUP_SIZE) if len(x) > GROUP_SIZE else x.sample(len(x)))

    # Split the Dataset into the Train and Validation Sets
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['n_ships'])
    train_df.drop('n_ships', axis=1, inplace=True)
    val_df.drop('n_ships', axis=1, inplace=True)
    
    # Merge with Masks DataFrame
    train_df = pd.merge(masks, train_df)
    val_df = pd.merge(masks, val_df)
    
    return train_df, val_df

