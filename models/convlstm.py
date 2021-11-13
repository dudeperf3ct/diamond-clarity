from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def convlstm(seq_len, img_height, img_width, channels):
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, 
                                data_format = "channels_last", 
                                input_shape=((seq_len, img_height, img_width, channels))))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(6, activation = "softmax"))

# if __name__ == '__main__':
#     m = convlstm(6, 384, 384, 3)
#     print(m.summary())