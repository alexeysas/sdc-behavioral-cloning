import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random

def load_samples(file_name, test_size=0.2, ignore_threshold=0.1, remove_probability=0.7):
    """ Load collected data reference and create train and validation set randomly shuffled
    ignore_threshold - angles less than this value are ignored
    and not added to the dataset with probability: remove_probability
    """
    lines = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            mesurement = float(line[3])
            if (np.abs(mesurement) > ignore_threshold) or (np.random.rand() > remove_probability):
                lines.append(line)
    train_samples, validation_samples = train_test_split(lines, test_size=test_size)
    return train_samples, validation_samples

def load_image(file_name):
    """ Load image from disk """
    image = cv2.imread(file_name)
    return image

#def prepare_image(image):
#    """Load image from disk"""
#    vertices = np.array([[(0, 140), (0, 50), (320, 50), (320, 140)]])
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = region_of_interest(image, vertices)
#    return image
#    img_out = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#    img_out = cv2.bilateralFilter(image,15,75,75)
#
#    return img_out

def get_augumented_images(line, get_image=load_image, 
        use_additional_cameras=False, correction=0.25):
    """ Create augumented images based on input image and angle,
    base version just flips image and add one additional image.
    Parameters:
    get_image - function to load image
    use_additional_cameras - indicates if we need to use images from
        right and left cameras in addition to central image
    correction - left and right camera angle adjustment using angle + correction
        or angle - correction
    """
    images = []
    angles = []

    #get image and angle from CSV line
    angle = float(line[3])

    #check if we need to use additional cameras
    if use_additional_cameras:
        index = random.randint(0, 2)
        image = get_image(line[index])
        if index == 1:
            angle = angle + correction
        elif index == 2:
            angle = angle - correction
    else:
        image = get_image(line[0])

    if random.randint(0, 1) == 1:
        #add base image to the resulting set
        images.append(image)
        angles.append(angle)
    else:
        #add flipped image to the resulting set
        images.append(cv2.flip(image, 1))
        angles.append(-angle)

    return images, angles

def generator(samples, batch_size, get_image, use_additional_cameras=False, correction=0.25):
    """ Generator based image retreival in case we have a lot of images which are not fit to memory
    get_image - function to load image
    use_additional_cameras - indicates if we need to use images from
        right and left cameras in addition to central image
    correction - left and right camera angle adjustment using angle + correction
        or angle - correction
    """
    num_samples = len(samples)
    internal_batch_size = batch_size

    while 1:
        #iterate over batches
        for offset in range(0, num_samples, internal_batch_size):
            batch_samples = samples[offset : offset + internal_batch_size]

            images = []
            angles = []

            #iterate over lines in batch
            for line in batch_samples:
                #extract data from the CSV line:
                new_images, new_angles = get_augumented_images(line, get_image=get_image,
                            use_additional_cameras=use_additional_cameras, correction=correction)

                images.extend(new_images)
                angles.extend(new_angles)

            x = np.array(images)
            y = np.array(angles)
            yield shuffle(x, y)

def load_image_data(samples, get_image=load_image,
        use_additional_cameras=False, correction=0.25):
    """ Load and preprocess image and angles data
    get_image - function to load image
    use_additional_cameras - indicates if we need to use images from
        right and left cameras in addition to central image
    correction - left and right camera angle adjustment using angle + correction
        or angle - correction
    """
    images = []
    angles = []

    for line in samples:
        #extract data from the CSV line:
        new_images, new_angles = get_augumented_images(line, get_image=get_image,
                    use_additional_cameras=use_additional_cameras, correction=correction)
        images.extend(new_images)
        angles.extend(new_angles)
    return np.array(images), np.array(angles)


def get_base_model():
    """ Create base model template with pre-processing layers """
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (20, 20)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    return model

def create_lenet_model():
    """ Build LeNeT model """
    model = get_base_model()
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.8))
    model.add(Dense(84))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def create_nvidia_model():
    """ Build NVIDIA model archirecture """
    model = get_base_model()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def show_angles_hist(angles):
    """ Utility method to show collected angles distribution """
    xbins = 20
    plt.hist(angles, bins=xbins, color='blue')
    plt.show()

def show_train_and_validation_loss(history_object):
    """ Utility method to display model loss  """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main_generator():
    """ preprocess image data and train model """
    batch_size = 64
    epochs = 15

    csv_log = "data/driving_log.csv"
    train_data, validation_data = load_samples(csv_log, 0.2, 0.2)

    train_generator = generator(train_data, batch_size=batch_size, get_image=load_image)
    validation_generator = generator(validation_data, batch_size=batch_size, get_image=load_image)

    model = create_nvidia_model()

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_data),
        validation_data=validation_generator, validation_steps=len(validation_data), epochs=epochs, verbose=1)

    model.save('model.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main():
    """ preprocess image data and train model """
    batch_size = 32
    epochs = 10

    csv_log = "data/driving_log.csv"
    train_data, validation_data = load_samples(csv_log, 0.2, 0.1, remove_probability=0.7)

    print("Loading image data...")

    train_x, train_y = load_image_data(train_data, use_additional_cameras=True, correction=0.3)
    validation_x, validation_y = load_image_data(validation_data, use_additional_cameras=False)

    #show_angles_hist(train_y)
    model = create_nvidia_model()

    history_object = model.fit(train_x, train_y, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_data = (validation_x, validation_y))

    model.save('model.h5')

    #show_train_and_validation_loss(history_object)
    pass

main()
#main_generator()

