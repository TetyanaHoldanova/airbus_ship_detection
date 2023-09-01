from keras import models, layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from image_segmentation.config import *
from image_segmentation.utils.metrics import dice_coef, dice_p_bce
from image_segmentation.utils.generator import data_generator, augmented_generator


class Unet:

    def build(self, shape=(H, W, 3)):
        '''
        Built the model
        @param shape: tuple with the shape of the training set
        @return: keras.models.Models object
        '''
        input_layer = layers.Input(shape=shape)                           # create an input layer
        pp_in_layer = input_layer
        
        if NET_SCALING is not None:
            pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)     # The input layer passed is passed through
                                                                         #  an Average Pooling 2D layer if NET_SCALING is specified
        
        # Contracting path
        conv1 = layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(pp_in_layer)
        conv1 = layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(conv1)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        
        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (pool1)
        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (conv2)
        pool2 = layers.MaxPooling2D((2, 2)) (conv2)
        
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (pool2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv3)
        pool3 = layers.MaxPooling2D((2, 2)) (conv3)
        
        conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (pool3)
        conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2)) (conv4)

        #Bottleneck
        conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (pool4)
        conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (conv5)
        
        # Expansive path
        up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv5)
        up6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (up6)
        conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (conv6)

        up7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv6)
        up7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (up7)
        conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (conv7)
        
        up8 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv7)
        up8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (up8)
        conv8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (conv8)

        up9 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (conv8)
        up9 = layers.concatenate([up9, conv1], axis=3)
        conv9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (up9)
        conv9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (conv9)

        # Output of the model
        conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        
        #  Upsample the output if NET_SCALING is specified
        if NET_SCALING is not None:
            conv10 = layers.UpSampling2D(NET_SCALING)(conv10)
        
        return models.Model([input_layer], [conv10])
    

    def fit(self, model, train_df, val_x, val_y, image_generator, masks_generator):
        '''
        Training the model
        @param model: keras.models.Models object
        @param train_df: training DataFrame
        @param val_x: validation DataFrame
        @param val_y: validation DataFrame
        @param image_generator: keras.preprocessing.image.ImageDataGenerator object
        @param masks_generator: : keras.preprocessing.image.ImageDataGenerator object
        @return: loss_history
        '''
        weight_path = "../assets/model/model_weights.best.hdf5"
        # Uses the ModelCheckpoint callback to save the best model 
        # weights based on the validation dice coefficient
        checkpoint = ModelCheckpoint(
            weight_path,
            monitor = 'val_dice_coeff',
            verbose=1,
            save_best_only = True,
            mode = 'max',
            save_weights_only = True)

        # The ReduceLROnPlateau callback is used to reduce the 
        # learning rate when the validation dice coefficient plateaus
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor = 'val_dice_coef',
            factor = 0.5,
            patience = 3, 
            mode = 'max', 
            cooldown = 2, 
            min_lr = 1e-6)

        # The EarlyStopping callback is used to stop training if the validation
        #  dice coefficient does not improve after a certain number of epochs
        early = EarlyStopping(
            monitor = "val_dice_coef",
            mode = "max",
            patience = 10)

        callbacks = [checkpoint, early, reduce_lr_on_plateau]
        
        # Compile with Adam optimizer
        model.compile(optimizer = Adam(lr = 1e-3), loss = dice_p_bce, metrics = [dice_coef])
        
        step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
        
        # Creating the augmented generator
        aug_gen = augmented_generator(data_generator(train_df), image_generator, masks_generator)
        
        # Train the model using the augmented generator
        loss_history = [model.fit_generator(aug_gen,
                                    steps_per_epoch=step_count,
                                    epochs=N_EPOCHS,
                                    validation_data=(val_x, val_y),
                                    callbacks=callbacks,
                                    workers=1)] 
        
        return loss_history
    

