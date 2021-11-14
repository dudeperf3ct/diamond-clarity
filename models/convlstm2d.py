from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def convlstm(seq_len, img_height, img_width, channels):
    model = Sequential()
    # input : (batch_size, num_frames, width, height, channels)
    model.add(layers.ConvLSTM2D(filters = 32, kernel_size = (5, 5), return_sequences = True, 
                                activation='relu', padding='same', 
                                input_shape=((seq_len, img_width, img_height, channels))))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 32, kernel_size = (3, 3), return_sequences = True, 
                                activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True, 
                                activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = True, 
                                activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 128, kernel_size = (3, 3), return_sequences = True, 
                                activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 128, kernel_size = (3, 3), return_sequences = True, 
                                activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 512, kernel_size = (3, 3), return_sequences = True, 
                                activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.ConvLSTM2D(filters = 512, kernel_size = (3, 3), return_sequences = True, 
                                activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation = "sigmoid"))
    return model

# if __name__ == '__main__':
#     m = convlstm(6, 384, 384, 3)
#     print(m.summary())