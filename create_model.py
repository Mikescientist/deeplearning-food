import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers

# dimensions of our images.
img_width, img_height = 225, 225

top_model_weights_path = '../Python/bottleneck_fc_model.h5'  # choose where you want to save the weights
train_data_dir = '../Python/data/train/'                # check these are consistent with create_dataset() directory
validation_data_dir = '../Python/data/validation/'
nb_train_samples = 504
nb_validation_samples = 300
epochs = 200
batch_size = 12


def save_bottlebeck_features():
    # Augment training data and normalise both to be in [0,1] 
    datagen1 = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.1,
        vertical_flip=True,
        horizontal_flip=True)
    datagen2 = ImageDataGenerator(rescale=1. / 255)

    # Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    # Save features
    generator = datagen2.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen2.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * int(252) + [1] * int(252))
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * int(150) + [1] * int(150))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification & Adam's optimiser
    Adam = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-08, decay=0.001)
    model.compile(Adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

# Classify new image with model
def predict_image(file):
    model = applications.VGG16(include_top=False, weights='imagenet')
    x = image.load_img(file, target_size=(img_width, img_height))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    model = Sequential()
    model.add(Flatten(input_shape=array.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights(top_model_weights_path)
    class_predicted = model.predict_classes(array)
    if class_predicted == 1:
        print("sushi")
    else:
        print("sandwich")

