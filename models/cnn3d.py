from tensorflow.keras import layers
from keras.models import Sequential

def cnn3d(width=384, height=384, depth=6, channels=3):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    model.add(layers.Conv3D(32, kernel_size=(11, 11, 1), input_shape=(width, height, depth, channels)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv3D(64, (7, 7, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(64, (5, 5, 1), (2, 2, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(128, (3, 3, 1), (2, 2, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 1), (2, 2, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# if __name__ == '__main__':
#     m = cnn3d(384, 284, 6, 3)
#     print(m.summary())
