import cv2
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split

EPOCHS = 200
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        pass
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print('data loaded')
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    x_train=tf.constant(x_train)
    y_train=tf.constant(y_train)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    with tf.device('GPU'):
        model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    sep=os.sep
    images=[]
    labels=[]
    for directory in range(NUM_CATEGORIES):
        for file in os.listdir('gtsrb'+sep+str(directory)):
            img=cv2.imread('gtsrb'+sep+str(directory)+sep+file)
            img.resize((IMG_HEIGHT,IMG_WIDTH,3))
            img=img/255.0
            images.append(img)
            labels.append(directory)
    return images,labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    '''
    # Model 1
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3,3),
        strides=(1,1),
        input_shape=(30,30,3),
        activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=(3,3)))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128,
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64,
                                    activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units=43,activation='softmax'))
    '''
    
    
   
    # Model 2
    model=None
    
    from keras.layers import Dense, Dropout, Conv2D ,Input,MaxPooling2D,Flatten
    input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
    model=tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))    
    
    
    
    model.add(Flatten())

    model.add(Dense(512,activation='relu',kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu',kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(Dense(43,activation='relu',kernel_regularizer='l2'))
    model.add(Dropout(0.15))
     
    
    model.add(Dense(NUM_CATEGORIES,activation='softmax'))
    
    
    
    
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics='accuracy',
        )
    return model
    
    
        


if __name__ == "__main__":
    main()
