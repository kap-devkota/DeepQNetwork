from tensorflow import keras


def get_cnn(final_layer_hidden_size):
    """
    Gets a convolutional neural network that will approximate the Q function.
    :param final_layer_hidden_size: The number of neurons in the last layer.
    :return: The CNN.
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(16,
                            (8, 8),
                            strides=(4, 4),
                            padding='same',
                            input_shape=(84, 84, 4), activation='relu'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32,
                            (4, 4),
                            strides=(2, 2),
                            padding='same',
                            activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(final_layer_hidden_size)
    ])

    model.compile(optimizer='Adam', loss=keras.losses.mean_squared_error)
    return model
