import random
import numpy as np
import pandas
import json
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.core import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.utils.np_utils import to_categorical

from utils import imageUtils

#import image utils and set the image input shape
image_utils = imageUtils()
im_x = image_utils.im_x
im_y = image_utils.im_y
im_z = image_utils.im_z

def get_model():
    """
        Defines the CNN model architecture and returns the model.
        The architecture is the same as I developed for project 2
        https://github.com/neerajdixit/Traffic-Sign-classifier-with-Deep-Learning
        with an additional normalization layer in front and
        a final fully connected layer of size 1 since we need one output.
    """

    # Create a Keras sequential model
    model = Sequential()
    #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    # Add a normalization layer to normalize between -0.5 and 0.5.
    model.add(Lambda(lambda x: x / 255. - .5,input_shape=(im_x,im_y,im_z), name='norm'))
    # Add a convolution layer with Input = 32x32x3. Output = 30x30x6. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(6, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv1'))
    # Add a convolution layer with Input = 30x30x6. Output = 28x28x9. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(9, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv2'))
    # Add Pooling layer with Input = 28x28x9. Output = 14x14x9. 2x2 kernel, Strides 2 and VALID padding
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='pool1'))
    # Add a convolution layer with Input 14x14x9. Output = 12x12x12. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(12, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv3'))
    # Add a convolution layer with Input = 30x30x6. Output = 28x28x9. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv4'))
    # Add Pooling layer with Input = 10x10x16. Output = 5x5x16. 2x2 kernel, Strides 2 and VALID padding
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='pool2'))
    # Flatten. Input = 5x5x16. Output = 400.
    model.add(Flatten(name='flat1'))
    # Add dropout layer with 0.2  
    model.add(Dropout(0.2, name='dropout1'))
    # Add Fully Connected layer. Input = 400. Output = 220
    # Perform RELU activation 
    model.add(Dense(220, activation='relu', name='fc1'))
    # Add Fully Connected layer. Input = 220. Output = 43
    # Perform RELU activation 
    model.add(Dense(43, activation='relu', name='fc2'))
    # Add Fully Connected layer. Input = 43. Output = 5
    # Perform RELU activation 
    model.add(Dense(5, name='fc3'))
    # Configure the model for training with Adam optimizer
    # "mean squared error" loss objective and accuracy metrics
    # Learning rate of 0.001 was chosen because this gave best performance after testing other values
    model.compile(optimizer=Adam(lr=0.001), loss="mse", metrics=['accuracy'])
    return model

def data_generator(data_path, X_train, y_train, batch_size):
    """
        Data generator for kera fit_generator
    """
    while True:
        # Select batch_size random indices from the image name array
        indices = np.random.randint(len(X_train),size=batch_size)
        # Get the corresponding steering angles
        y = y_train[indices]
        # Create empty numpy array of batch size for images
        x=np.zeros((batch_size, im_x, im_y, im_z))
        for i in range(batch_size):
            # Read the image from data path using open cv
            sample = X_train[indices[i]]
            img = cv2.imread(data_path+sample[0].strip())
            # pre process image and add to array
            x[i] = image_utils.pre_process_image(img, sample[1], sample[2], sample[3], sample[4])
        yield (x, y)

def setup_data(path):
    """
        Reads the log file from data_path and creates the data used by generators.
        Takes in the log file location as parameter
    """
    # Read the csv file using pandas
    train_data = pandas.read_csv(path+'train.csv', delim_whitespace=True, header = None)
    # frame, xmin, ymin, xmax, ymax, occluded, label
    X_train = np.dstack((np.array(train_data[0]),
                         np.array(train_data[1]),
                         np.array(train_data[2]),
                         np.array(train_data[3]),
                         np.array(train_data[4])))
    y_train = np.array(train_data[6])
    y_train = y_train.reshape(1,y_train.shape[0])
    # binarize the labels
    # car = 0,truck = 1,biker = 2,pedestrian = 3,trafficLight = 4
    yy_train = to_categorical(y_train[0])
    # Shuffle the data.
    # Important before dividing the test data into training and validation sets. To avoid same data on multiple runs.
    X_train, y_train = shuffle(X_train[0], yy_train)
    # Split the test data in training and validation sets.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train , test_size=0.3, random_state=0)
    print("Number of training examples =", len(X_train))
    print("Number of validation examples =", len(X_validation))
    return X_train, X_validation, y_train, y_validation


######## Processing ########
data_path = './object-dataset/'
# Setup data from drive log csv
X_train, X_validation, y_train, y_validation = setup_data('./')

#Get model and print summary.
model = get_model()
print(model.summary())

# Set batch size and epocs
per_epoch_samples=50000#len(X_train)
gen_batch_size=500
epochs=50

# Fit the data on model and validate using data generators. 
model.fit_generator(data_generator(data_path, 
                                    X_train, 
                                    y_train, 
                                    gen_batch_size),
                    samples_per_epoch=per_epoch_samples, 
                    nb_epoch=epochs,
                    validation_data=data_generator(data_path, 
                        X_validation, 
                        y_validation, 
                        gen_batch_size),
                    nb_val_samples=len(X_validation))

# Save the model and weights
print("Saving model weights and configuration file...")
model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
print("model saved...")