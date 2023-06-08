import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Calculates the Dice coefficient  to measure the similarity 
    between the predicted and true segmentation masks
    @param y_true: np.array 
    @param y_pred: np.array
    @return: np.array of the means of the Dice coefficient values computed for each batch element
    '''
    # Calculating the intersection between y_true and y_pred
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    # Calculating the union of y_true and y_pred
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    # Returns the mean of the Dice coefficient values computed for each batch element
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    '''
    Combines the Dice coefficient and binary cross-entropy loss
    @param in_gt: ground truth
    @param in_pred: predicted values
    @return: combination of binary cross-entropy loss and Dice coefficient
    '''
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)