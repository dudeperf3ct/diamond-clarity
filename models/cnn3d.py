from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import AdaptiveAveragePooling3D

def cnn3d(width=384, height=384, depth=6, channels=3):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 1), input_shape=(width, height, depth, channels)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(64, (3, 3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(128, (3, 3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(512, (3, 3, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(AdaptiveAveragePooling3D(output_size=(1, 1, 6)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# if __name__ == '__main__':
#     m = cnn3d(384, 384, 6, 3)
#     print(m.summary())
