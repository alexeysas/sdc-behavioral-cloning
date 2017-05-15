import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_samples(file_name, test_size=0.2, ignore_threshold=0.1):
    """Load collected data reference and create train and validation set randomly shuffled"""
    lines = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            mesurement = float(line[3])
            if (np.abs(mesurement) > ignore_threshold) or (np.random.rand() > 0.7):
                lines.append(line)
    train_samples, validation_samples = train_test_split(lines, test_size=test_size, random_state = 5)
    return train_samples, validation_samples


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def load_image(file_name):
    """Load image from disk"""
    image = cv2.imread(file_name)
    return image


def prepare_image(image):
    """Load image from disk"""    
    vertices = np.array([[(0, 130), (0, 70), (320, 70), (320, 130)]])
    image = region_of_interest(image, vertices)
    return image
    img_out = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_out = cv2.bilateralFilter(image,15,75,75)
    
    return img_out

    

def generator(samples, batch_size, get_image):
    num_samples = len(samples)
    factor = 2 # we have augumented images
    internal_batch_size = batch_size // factor
    while 1:
        for offset in range(0, num_samples, internal_batch_size):
            batch_samples = samples[offset : offset + internal_batch_size]

            images = []
            angles = []

            for line in batch_samples:
                image = get_image(line[0])
                mesurement = float(line[3])

                images.append(image)
                angles.append(mesurement)

                images.append(cv2.flip(image, 1))
                angles.append(-mesurement)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)

def load_image_data(samples, get_image=load_image):
    """ preprocess image and mesurement data """
    images = []
    angles = []

    for line in samples:
        image = get_image(line[0])
        #image = prepare_image(image)
        mesurement = float(line[3])
        images.append(image)
        angles.append(mesurement)
        images.append(cv2.flip(image, 1))
        angles.append(-mesurement)

        correction_b = 0.1
        correction_a = 1

        #steering_left = correction_a * mesurement + correction_b
        #steering_right = correction_a * mesurement - correction_b

        #image = get_image(line[1])
        #images.append(image)
        #angles.append(steering_left)
        #images.append(cv2.flip(image, 1))
        #angles.append(-steering_left)

        #image = get_image(line[2])
        #images.append(image)
        #angles.append(steering_right)
        #mages.append(cv2.flip(image, 1))
        #angles.append(-steering_right)

    return np.array(images), np.array(angles)

def create_model():
    """ build model """
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.8))
    model.add(Dense(50))
    model.add(Dropout(0.8))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def main_generator():
    """ preprocess image data and train model """
    batch_size = 64

    csv_log = "data/driving_log.csv"
    train_data, validation_data = load_samples(csv_log, 0.2, 0.2)

    train_generator = generator(train_data, batch_size=batch_size, get_image = get_image_func)
    validation_generator = generator(validation_data, batch_size=batch_size, get_image = get_image_func)

    model = create_model()

    history_object = model.fit(train_generator, steps_per_epoch=len(train_data),
        validation_data=validation_generator, validation_steps=len(validation_data), epochs=20, verbose=1)

    model.save('model.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    pass

def main():
    """ preprocess image data and train model """
    batch_size = 32
    epochs = 10

    csv_log = "data/driving_log.csv"
    train_data, validation_data = load_samples(csv_log, 0.1, 0.3)

    #image = load_image(train_data[0][0])
    #plt.imshow(image)
    #plt.show()

    #image = prepare_image(image)
    #plt.imshow(image)
    #plt.show()
    #return
    print ("Load image data")

    train_x, train_y = load_image_data(train_data)
    validation_x, validation_y = load_image_data(validation_data)
    
    model = create_model()

    history_object = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data = (validation_x, validation_y))

    model.save('model.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    pass

main()
#main_generator()
