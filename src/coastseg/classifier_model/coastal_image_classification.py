"""
Mark Lundine
Good/bad coastal image classification
adapted from https://keras.io/examples/vision/image_classification_from_scratch/
"""
import os
import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import shutil
import matplotlib.pyplot as plt
    
def load_dataset(training_data_directory,
                 image_size,
                 batch_size):
    """
    loads in training data in training and validation sets
    inputs:
    training_data_directory (str): path to the training data
    returns:
    train_ds, val_ds
    """
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(training_data_directory,
                                                                        validation_split=0.2,
                                                                        color_mode="grayscale",
                                                                        subset="both",
                                                                        seed=1337,
                                                                        image_size=image_size,
                                                                        batch_size=batch_size
                                                                        )
    return train_ds, val_ds

def data_augmentation(images):
    """
    applies data augmentation to images
    """
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.5, 0.5),
        
    ]
    return images

def define_model(input_shape, num_classes=2):
    """
    Defines the classification model
    inputs:
    input_shape (tuple (xdim, ydim)): shape of images for model
    num_classes (int, optional): number of classes for the model
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = data_augmentation(inputs)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)

    return keras.Model(inputs, outputs)

def train_model(model,
                image_size,
                train_ds,
                val_ds,
                model_folder,
                epochs=100):
    """
    Trains the good/bad classification model
    inputs:
    model (keras model): this is defined in define_model()
    image_size (tuple, (xdim, ydim): dimensions of images for model
    train_ds: training dataset obtained from load_dataset
    val_ds: validation dataset obtained from load_dataset
    model_folder (str): path to save the model to
    epochs (int, optional): number of epochs to train for
    """
    model = define_model(input_shape=image_size + (1,), num_classes=2)

    ##this makes a plot of the model, you need pydot installed, 
    #keras.utils.plot_model(model, to_file=os.path.join(model_folder, 'model_graph.png'), show_shapes=True)
    
    ckpt_file = os.path.join(model_folder, "model_{epoch}.h5")
    epochs = 25
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
    callbacks = [keras.callbacks.ModelCheckpoint(ckpt_file),
                 early_stopping_callback
                 ]
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')
                           ]
                  )
    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_ds
                        )
    return model, history, ckpt_file

def sort_images(inference_df_path,
                output_folder,
                threshold=0.40):
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    """
    bad_dir = os.path.join(output_folder, 'bad')
    dirs = [output_folder, bad_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path) 
        if inference_df['model_scores'].iloc[i] < threshold:
            output_image_path = os.path.join(bad_dir, im_name)
            shutil.move(input_image_path, output_image_path)
        
def run_inference(path_to_model_ckpt,
                  path_to_inference_imgs,
                  output_folder,
                  result_path,
                  threshold):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = keras.models.load_model(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode='grayscale',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    df.to_csv(result_path,index=False)
    sort_images(result_path,
                output_folder,
                threshold=threshold)
    return result_path

    
def plot_history(history,
                 history_save_path):
    """
    This makes a plot of the loss curve
    inputs:
    history: history object from model.fit_generator
    history_save_path (str): path to save the plot to
    """
    plt.subplot(4,1,1)
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('Loss (BCE)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
    plt.subplot(4,1,2)
    plt.plot(history.history['acc'], color='b')
    plt.plot(history.history['val_acc'], color='r')
    plt.minorticks_on()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
    plt.subplot(4,1,3)
    plt.plot(history.history['precision'], color='b')
    plt.plot(history.history['val_precision'], color='r')
    plt.minorticks_on()
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')

    plt.subplot(4,1,4)
    plt.plot(history.history['recall'], color='b')
    plt.plot(history.history['val_recall'], color='r')
    plt.minorticks_on()
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    plt.tight_layout()
    plt.savefig(history_save_path, dpi=300)
    plt.close('all')
    
def training(path_to_training_data,
             output_folder,
             epochs=25):
    """
    Training the good/bad classification model
    inputs:
    path_to_training_data (str): path to the directory containing good and bad subdirectories
    output_folder (str): path to save outputs to
    epoch (int, optional): number of epochs to train for
    """
    ##Making new directories
    try:
        os.mkdir(output_folder)
    except:
        pass
    
    model_folder = os.path.join(output_folder, 'models')
    history_save_path = os.path.join(model_folder, 'history.png')
    try:
        os.mkdir(model_folder)
    except:
        pass

    ##Setting image size for model and loading datasets
    image_size = (128, 128)
    train_ds, val_ds = load_dataset(path_to_training_data,
                                    (128, 128),
                                    32)

    ##Defining and then training model
    model = define_model(input_shape=image_size + (1,), num_classes=2)
    model, history, ckpt_file = train_model(model,
                                            image_size,
                                            train_ds,
                                            val_ds,
                                            model_folder,
                                            epochs=epochs)
    best_ckpt_file = os.path.join(model_folder, 'best.h5')
    model.save(best_ckpt_file)

    ##Plotting and saving loss curve
    plot_history(history, history_save_path)
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = os.path.join(os.path.join(model_folder, 'history.csv'))
    hist_df.to_csv(hist_csv_file, index=False)    

    return best_ckpt_file
    
##"""
##Sample Train
##"""

# training(os.path.join(os.getcwd(),'sorted','train'),
#         os.getcwd(),
#         epochs=25)    


"""
Sample Call on Test Dataset
"""
##run_inference("""path/to/best.h5""",
##              """path/to/jpgs_directory""",
##              """path/to/output_directory""",
##              """path/to/output_directory/output.csv""",
##              0.1
##              )



    
